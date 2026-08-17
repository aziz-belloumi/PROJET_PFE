from difflib import SequenceMatcher
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from src.config import Config


class DatabaseConnection:
    def __init__(self, logger=None):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.engine = None
        self._connect()

    def _connect(self):
        try:
            self.engine = create_engine(
                Config.SQLALCHEMY_DATABASE_URI,
                poolclass=QueuePool,
                pool_pre_ping=True,
                pool_recycle=Config.DB_POOL_RECYCLE,
                pool_size=Config.DB_POOL_SIZE,
                max_overflow=Config.DB_MAX_OVERFLOW,
                echo=False,
            )
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            self.logger.info("MySQL connection established successfully.")
        except Exception as e:
            self.logger.error(f"MySQL connection error: {e}")
            raise

    def get_engine(self):
        return self.engine

    def is_connected(self) -> bool:
        """Verify that the database connection pool is active and reachable."""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def get_pool_status(self) -> dict:
        """Return current connection pool metrics."""
        if not self.engine or not hasattr(self.engine, "pool"):
            return {}
        pool = self.engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }

    # ============================================================
    # Query Execution Helpers (Transactional Safety)
    # ============================================================

    def execute_write(self, sql_query: str, params: dict | None = None) -> int:
        """
        Execute an INSERT / UPDATE / DELETE / DDL statement inside an atomic transaction.
        Automatically commits on success and rolls back on exception.
        """
        try:
            with self.engine.begin() as conn:
                result = conn.execute(text(sql_query), params or {})
                return result.rowcount
        except Exception as e:
            self.logger.error(f"SQL Write Error: {e}")
            raise

    def execute_query(self, sql_query: str, params: dict | None = None):
        """Backward-compatible alias for execute_write."""
        return self.execute_write(sql_query, params)

    def fetch_one(self, sql_query: str, params: dict | None = None):
        """Execute a SELECT query and return the first row, or None."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query), params or {})
                return result.fetchone()
        except Exception as e:
            self.logger.error(f"SQL Read Error (fetch_one): {e}")
            raise

    def fetch_all(self, sql_query: str, params: dict | None = None) -> list:
        """Execute a SELECT query and return all matching rows."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query), params or {})
                return result.fetchall()
        except Exception as e:
            self.logger.error(f"SQL Read Error (fetch_all): {e}")
            raise

    # ============================================================
    # Table Management
    # ============================================================

    RESULT_TABLES = [
        "article_topics",
        "article_entities",
        "entities",
        "articles_enriched",
        "qwen_article_topics",
        "qwen_articles_enriched",
        "benchmark_results",
        "qwen_benchmark_results",
    ]

    def drop_result_tables(self):
        try:
            with self.engine.begin() as conn:
                for table in self.RESULT_TABLES:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
            self.logger.info(f"Dropped {len(self.RESULT_TABLES)} enriched tables.")
        except Exception as e:
            self.logger.error(f"Error dropping tables: {e}")
            raise

    def create_result_tables(self):
        tables = [
            # 1) articles_enriched (unified LLM results)
            """
            CREATE TABLE IF NOT EXISTS articles_enriched (
                article_id      BIGINT NOT NULL,
                language        VARCHAR(10),
                sentiment_label VARCHAR(10),
                sentiment_score FLOAT,

                PRIMARY KEY (article_id),
                INDEX idx_lang (language)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,

            # 2) entities dictionary
            """
            CREATE TABLE IF NOT EXISTS entities (
                entity_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                entity_name VARCHAR(1024) NOT NULL,
                entity_type VARCHAR(20) NOT NULL,
                normalized_name VARCHAR(512) NOT NULL,
                frequency INT DEFAULT 0,

                UNIQUE KEY uq_entity (entity_type, normalized_name),
                INDEX idx_type (entity_type),
                INDEX idx_norm (normalized_name),
                INDEX idx_freq (frequency)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,

            # 3) article_entities link (with model_version as number)
            """
            CREATE TABLE IF NOT EXISTS article_entities (
                article_id BIGINT NOT NULL,
                entity_id BIGINT NOT NULL,
                model_version TINYINT NOT NULL,
                confidence_score FLOAT,

                PRIMARY KEY (article_id, entity_id, model_version),
                INDEX idx_article (article_id),
                INDEX idx_entity (entity_id),
                INDEX idx_model (model_version),

                CONSTRAINT fk_article_entities_entity
                  FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
                  ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,

            # 4) article_topics (unified results)
            """
            CREATE TABLE IF NOT EXISTS article_topics (
                article_id  BIGINT NOT NULL,
                topic_label VARCHAR(100) NOT NULL,
                topic_score FLOAT,

                PRIMARY KEY (article_id),
                INDEX idx_topic (topic_label)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,

            # 5) dedicated Qwen results tables
            """
            CREATE TABLE IF NOT EXISTS qwen_article_topics (
                article_id  BIGINT NOT NULL,
                topic_label VARCHAR(100) NOT NULL,
                topic_score FLOAT,

                PRIMARY KEY (article_id),
                INDEX idx_qwen_topic (topic_label)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,

            """
            CREATE TABLE IF NOT EXISTS qwen_articles_enriched (
                article_id      BIGINT NOT NULL,
                language        VARCHAR(10),
                sentiment_label VARCHAR(10),
                sentiment_score FLOAT,

                PRIMARY KEY (article_id),
                INDEX idx_qwen_lang (language)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,

            # 6) standard benchmark results
            """
            CREATE TABLE IF NOT EXISTS benchmark_results (
                article_id BIGINT NOT NULL,
                task VARCHAR(20) NOT NULL,
                model_id TINYINT NOT NULL,
                language VARCHAR(10),
                device VARCHAR(10) NOT NULL DEFAULT 'GPU',
                total_inf_time_sec DOUBLE,
                avg_ms_per_doc DOUBLE,
                peak_gpu_mb DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (article_id, task, model_id, language, device),
                INDEX idx_benchmark_results_task (task),
                INDEX idx_benchmark_results_model_id (model_id),
                INDEX idx_benchmark_results_article (article_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,

            # 7) Qwen benchmark results
            """
            CREATE TABLE IF NOT EXISTS qwen_benchmark_results (
                article_id BIGINT NOT NULL,
                task VARCHAR(20) NOT NULL,
                model_id TINYINT NOT NULL,
                language VARCHAR(10),
                device VARCHAR(10) NOT NULL DEFAULT 'GPU',
                total_inf_time_sec DOUBLE,
                avg_ms_per_doc DOUBLE,
                peak_gpu_mb DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                PRIMARY KEY (article_id, task, model_id, language, device),
                INDEX idx_qwen_benchmark_results_task (task),
                INDEX idx_qwen_benchmark_results_model_id (model_id),
                INDEX idx_qwen_benchmark_results_article (article_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
        ]

        # Cross-version safe schema migrations (MySQL 5.7, 8.0+, MariaDB)
        def _safe_add_column(conn, table: str, col: str, col_type: str):
            try:
                res = conn.execute(text(f"SHOW COLUMNS FROM {table} LIKE '{col}'")).fetchone()
                if not res:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                    conn.commit()
            except Exception as e:
                self.logger.debug(f"Migration note adding {table}.{col}: {e}")

        def _safe_drop_column(conn, table: str, col: str):
            try:
                res = conn.execute(text(f"SHOW COLUMNS FROM {table} LIKE '{col}'")).fetchone()
                if res:
                    conn.execute(text(f"ALTER TABLE {table} DROP COLUMN {col}"))
                    conn.commit()
            except Exception as e:
                self.logger.debug(f"Migration note dropping {table}.{col}: {e}")

        def _safe_drop_index(conn, table: str, idx: str):
            try:
                res = conn.execute(text(f"SHOW INDEX FROM {table} WHERE Key_name = '{idx}'")).fetchone()
                if res:
                    conn.execute(text(f"ALTER TABLE {table} DROP INDEX {idx}"))
                    conn.commit()
            except Exception as e:
                self.logger.debug(f"Migration note dropping index {table}.{idx}: {e}")

        try:
            with self.engine.connect() as conn:
                for sql in tables:
                    try:
                        conn.execute(text(sql))
                        conn.commit()
                    except Exception as e:
                        self.logger.error(f"Error executing table creation: {e}")

                # Safe migrations
                _safe_add_column(conn, "articles_enriched", "sentiment_score", "FLOAT")
                _safe_add_column(conn, "article_topics", "topic_score", "FLOAT")
                _safe_add_column(conn, "qwen_article_topics", "topic_score", "FLOAT")
                _safe_add_column(conn, "qwen_articles_enriched", "sentiment_score", "FLOAT")

                _safe_drop_column(conn, "articles_enriched", "gpu_time_ner")
                _safe_drop_column(conn, "articles_enriched", "gpu_time_sentiment")
                _safe_drop_column(conn, "articles_enriched", "gpu_time_topic")
                _safe_drop_column(conn, "qwen_articles_enriched", "gpu_time_sentiment")
                _safe_drop_column(conn, "qwen_articles_enriched", "gpu_time_topic")
                _safe_drop_column(conn, "benchmark_results", "load_time_sec")
                _safe_drop_column(conn, "qwen_benchmark_results", "load_time_sec")

                _safe_drop_index(conn, "article_topics", "idx_article")
                _safe_drop_index(conn, "qwen_article_topics", "idx_qwen_article")

            self.logger.info("Enriched tables created successfully.")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    def init_result_tables(self):
        self.create_result_tables()

    # ============================================================
    # Data Insert / Upsert Methods
    # ============================================================

    def upsert_benchmark_result(
        self,
        article_id: int,
        task: str,
        model_id: int,
        language: str | None = None,
        device: str = "GPU",
        total_inf_time_sec: float | None = None,
        avg_ms_per_doc: float | None = None,
        peak_gpu_mb: float | None = None,
    ):
        sql = """
        INSERT INTO benchmark_results (
            article_id, task, model_id, language, device,
            total_inf_time_sec, avg_ms_per_doc, peak_gpu_mb
        ) VALUES (
            :article_id, :task, :model_id, :language, :device,
            :total_inf_time_sec, :avg_ms_per_doc, :peak_gpu_mb
        )
        ON DUPLICATE KEY UPDATE
            total_inf_time_sec = VALUES(total_inf_time_sec),
            avg_ms_per_doc = VALUES(avg_ms_per_doc),
            peak_gpu_mb = VALUES(peak_gpu_mb)
        """
        self.execute_write(sql, {
            "article_id": int(article_id),
            "task": str(task),
            "model_id": int(model_id),
            "language": language,
            "device": str(device),
            "total_inf_time_sec": float(total_inf_time_sec) if total_inf_time_sec is not None else None,
            "avg_ms_per_doc": float(avg_ms_per_doc) if avg_ms_per_doc is not None else None,
            "peak_gpu_mb": float(peak_gpu_mb) if peak_gpu_mb is not None else None,
        })

    def upsert_qwen_benchmark_result(
        self,
        article_id: int,
        task: str,
        model_id: int,
        language: str | None = None,
        device: str = "GPU",
        total_inf_time_sec: float | None = None,
        avg_ms_per_doc: float | None = None,
        peak_gpu_mb: float | None = None,
    ):
        sql = """
        INSERT INTO qwen_benchmark_results (
            article_id, task, model_id, language, device,
            total_inf_time_sec, avg_ms_per_doc, peak_gpu_mb
        ) VALUES (
            :article_id, :task, :model_id, :language, :device,
            :total_inf_time_sec, :avg_ms_per_doc, :peak_gpu_mb
        )
        ON DUPLICATE KEY UPDATE
            total_inf_time_sec = VALUES(total_inf_time_sec),
            avg_ms_per_doc = VALUES(avg_ms_per_doc),
            peak_gpu_mb = VALUES(peak_gpu_mb)
        """
        self.execute_write(sql, {
            "article_id": int(article_id),
            "task": str(task),
            "model_id": int(model_id),
            "language": language,
            "device": str(device),
            "total_inf_time_sec": float(total_inf_time_sec) if total_inf_time_sec is not None else None,
            "avg_ms_per_doc": float(avg_ms_per_doc) if avg_ms_per_doc is not None else None,
            "peak_gpu_mb": float(peak_gpu_mb) if peak_gpu_mb is not None else None,
        })

    def upsert_qwen_article_topic(
        self,
        article_id: int,
        topic_label: str,
        confidence_score: float | None = None,
    ):
        sql = """
        INSERT INTO qwen_article_topics (article_id, topic_label, topic_score)
        VALUES (:aid, :lab, :score)
        ON DUPLICATE KEY UPDATE
            topic_label = VALUES(topic_label),
            topic_score = COALESCE(VALUES(topic_score), topic_score)
        """
        self.execute_write(sql, {
            "aid": int(article_id),
            "lab": str(topic_label),
            "score": float(confidence_score) if confidence_score is not None else None,
        })

    def upsert_qwen_articles_enriched(
        self,
        article_id: int,
        language: str | None = None,
        sentiment_label: str | None = None,
        sentiment_score: float | None = None,
    ):
        sql = """
        INSERT INTO qwen_articles_enriched (
            article_id, language, sentiment_label, sentiment_score
        ) VALUES (
            :aid, :lang, :sent, :score
        )
        ON DUPLICATE KEY UPDATE
            language = COALESCE(VALUES(language), language),
            sentiment_label = COALESCE(VALUES(sentiment_label), sentiment_label),
            sentiment_score = COALESCE(VALUES(sentiment_score), sentiment_score)
        """
        self.execute_write(sql, {
            "aid": int(article_id),
            "lang": language,
            "sent": sentiment_label,
            "score": float(sentiment_score) if sentiment_score is not None else None,
        })

    def upsert_articles_enriched(
        self,
        article_id: int,
        language: str | None = None,
        sentiment_label: str | None = None,
        sentiment_score: float | None = None,
    ):
        sql = """
        INSERT INTO articles_enriched (
            article_id, language,
            sentiment_label, sentiment_score
        ) VALUES (
            :aid, :lang,
            :s_lbl, :s_score
        )
        ON DUPLICATE KEY UPDATE
            language = COALESCE(VALUES(language), language),
            sentiment_label = COALESCE(VALUES(sentiment_label), sentiment_label),
            sentiment_score  = COALESCE(VALUES(sentiment_score),  sentiment_score)
        """
        self.execute_write(sql, {
            "aid":     int(article_id),
            "lang":    language,
            "s_lbl":   sentiment_label,
            "s_score": float(sentiment_score) if sentiment_score is not None else None,
        })

    def find_similar_entity(
        self,
        normalized_name: str,
        entity_type: str,
        threshold: float = 0.9,
    ) -> int | None:
        """
        Find an existing entity by exact match or bounded fuzzy match.
        Uses LIMIT 100 on fuzzy candidate search to prevent unbounded table scans.
        """
        if not normalized_name or not entity_type:
            return None

        # Enforce length limits mathematically tied to DB schema
        normalized_name = normalized_name[:512]
        entity_type = entity_type[:20]

        # 1. Exact match
        sql_exact = "SELECT entity_id FROM entities WHERE entity_type = :typ AND normalized_name = :norm LIMIT 1"
        row = self.fetch_one(sql_exact, {"typ": entity_type, "norm": normalized_name})
        if row:
            return int(row[0])

        # 2. Fuzzy fallback for sufficiently long names
        if len(normalized_name) < 5:
            return None

        prefix = normalized_name[:3]
        sql_fuzzy = """
        SELECT entity_id, normalized_name FROM entities
        WHERE entity_type = :typ AND normalized_name LIKE :pref
        LIMIT 100
        """
        rows = self.fetch_all(sql_fuzzy, {"typ": entity_type, "pref": f"{prefix}%"})

        for eid, existing_name in rows:
            ratio = SequenceMatcher(None, normalized_name, existing_name).ratio()
            if ratio >= threshold:
                return int(eid)

        return None

    def upsert_entity(self, entity_name: str, entity_type: str, normalized_name: str) -> int:
        """
        Insert a new entity or return the existing entity_id.
        Employs atomic LAST_INSERT_ID(entity_id) on duplicate key to eliminate race conditions.
        """
        entity_name = (entity_name or "")[:1024]
        entity_type = (entity_type or "")[:20]
        normalized_name = (normalized_name or "")[:512]

        if not normalized_name or not entity_type:
            return -1

        # 1. Check for existing exact/fuzzy match
        existing_id = self.find_similar_entity(normalized_name, entity_type)
        if existing_id:
            return existing_id

        # 2. Atomic insert with race-condition safety
        sql = """
        INSERT INTO entities (entity_name, entity_type, normalized_name)
        VALUES (:name, :typ, :norm)
        ON DUPLICATE KEY UPDATE entity_id = LAST_INSERT_ID(entity_id)
        """
        try:
            with self.engine.begin() as conn:
                conn.execute(text(sql), {
                    "name": entity_name,
                    "typ": entity_type,
                    "norm": normalized_name,
                })
                res = conn.execute(text("SELECT LAST_INSERT_ID()"))
                entity_id = int(res.scalar())
            return entity_id
        except Exception as e:
            self.logger.error(f"Error upserting entity '{normalized_name}': {e}")
            raise

    def update_entity_frequencies(self):
        """Update each entity's frequency count based on article_entities occurrences."""
        self.logger.info("Computing global entity frequencies...")
        sql = """
        UPDATE entities e
        JOIN (
            SELECT entity_id, COUNT(*) as cnt
            FROM article_entities
            GROUP BY entity_id
        ) stats ON e.entity_id = stats.entity_id
        SET e.frequency = stats.cnt
        """
        self.execute_write(sql)

    def upsert_article_entity(
        self,
        article_id: int,
        entity_id: int,
        model_version: int,
        confidence_score: float | None,
    ):
        """Insert or update an article-entity link with highest confidence score."""
        sql = """
        INSERT INTO article_entities (article_id, entity_id, model_version, confidence_score)
        VALUES (:aid, :eid, :mv, :conf)
        ON DUPLICATE KEY UPDATE
            confidence_score = GREATEST(COALESCE(confidence_score, 0), VALUES(confidence_score))
        """
        self.execute_write(sql, {
            "aid": int(article_id),
            "eid": int(entity_id),
            "mv": int(model_version),
            "conf": float(confidence_score) if confidence_score is not None else None,
        })

    def upsert_article_topic(
        self,
        article_id: int,
        topic_label: str,
        confidence_score: float | None = None,
    ):
        """Insert or update an article's topic assignment and confidence score."""
        sql = """
        INSERT INTO article_topics (article_id, topic_label, topic_score)
        VALUES (:aid, :lab, :score)
        ON DUPLICATE KEY UPDATE
            topic_label = VALUES(topic_label),
            topic_score = COALESCE(VALUES(topic_score), topic_score)
        """
        self.execute_write(sql, {
            "aid": int(article_id),
            "lab": str(topic_label),
            "score": float(confidence_score) if confidence_score is not None else None,
        })

    def close(self):
        """Gracefully close and dispose the connection pool."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection pool disposed.")
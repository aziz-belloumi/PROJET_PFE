from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from src.config import Config
import logging


class DatabaseConnection:
    def __init__(self, logger=None):
        self.logger = logger if logger else logging.getLogger(__name__)
        self.engine = None
        self.Session = None
        self._connect()

    def _connect(self):
        try:
            self.engine = create_engine(
                Config.SQLALCHEMY_DATABASE_URI,
                poolclass=QueuePool,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False,
            )
            with self.engine.connect() as conn:
                self.logger.info("MySQL connection established successfully.")
            self.Session = sessionmaker(bind=self.engine)
        except Exception as e:
            self.logger.error(f"MySQL connection error: {e}")
            raise

    def get_engine(self):
        return self.engine

    def execute_query(self, sql_query, params=None):
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query), params or {})
                conn.commit()
                return result
        except Exception as e:
            self.logger.error(f"SQL Error: {e}")
            raise

    # ============================================================
    # Table Management
    # ============================================================

    RESULT_TABLES = [
        "article_topics",
        "article_entities",
        "entities",
        "articles_enriched",
    ]

    def drop_result_tables(self):
        try:
            with self.engine.connect() as conn:
                for table in self.RESULT_TABLES:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                conn.commit()
            self.logger.info(f"Dropped {len(self.RESULT_TABLES)} enriched tables.")
        except Exception as e:
            self.logger.error(f"Error dropping tables: {e}")
            raise

    def create_result_tables(self):
        """
        Supervisor structure + agreed addition:
        - article_entities has model_version (0/1/2)
        No extra columns beyond what we agreed.
        """
        try:
            self.logger.info("Creating enriched tables...")
            with self.engine.connect() as conn:

                # 1) articles_enriched (rows multiplied by 3 via model_version)
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS articles_enriched (
                    article_id BIGINT NOT NULL,
                    model_version TINYINT NOT NULL,  -- 0=arabert, 1=camel, 2=mbert

                    language VARCHAR(10),

                    sentiment_label VARCHAR(10),
                    sentiment_score FLOAT,

                    dominant_topic VARCHAR(100),

                    cpu_processing_time BIGINT,
                    gpu_processing_time BIGINT,

                    PRIMARY KEY (article_id, model_version),
                    INDEX idx_lang (language),
                    INDEX idx_model (model_version)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                # 2) entities dictionary
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS entities (
                    entity_id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    entity_name VARCHAR(255) NOT NULL,
                    entity_type VARCHAR(20) NOT NULL,
                    normalized_name VARCHAR(255) NOT NULL,
                    frequency INT DEFAULT 0,

                    UNIQUE KEY uq_entity (entity_type, normalized_name),
                    INDEX idx_type (entity_type),
                    INDEX idx_norm (normalized_name),
                    INDEX idx_freq (frequency)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                # 3) article_entities link (with model_version as number)
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS article_entities (
                    article_id BIGINT NOT NULL,
                    entity_id BIGINT NOT NULL,
                    model_version TINYINT NOT NULL,  -- 0=arabert, 1=camel, 2=mbert
                    confidence_score FLOAT,

                    PRIMARY KEY (article_id, entity_id, model_version),
                    INDEX idx_article (article_id),
                    INDEX idx_entity (entity_id),
                    INDEX idx_model (model_version),

                    CONSTRAINT fk_article_entities_entity
                      FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
                      ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                # 4) article_topics (no rank)
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS article_topics (
                    article_id BIGINT NOT NULL,
                    topic_label VARCHAR(100) NOT NULL,
                    topic_score FLOAT,

                    PRIMARY KEY (article_id),
                    INDEX idx_article (article_id),
                    INDEX idx_topic (topic_label)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                conn.commit()

            self.logger.info("Enriched tables created successfully.")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    def init_result_tables(self):
        """Ensure enriched tables exist without dropping existing data."""
        self.create_result_tables()

    # ============================================================
    # Data Insert / Upsert Methods
    # ============================================================

    def upsert_articles_enriched(
        self,
        article_id: int,
        model_version: int,
        language: str | None = None,
        sentiment_label: str | None = None,
        sentiment_score: float | None = None,
        dominant_topic: str | None = None,
        cpu_processing_time: int | None = None,
        gpu_processing_time: int | None = None,
    ):
        sql = """
        INSERT INTO articles_enriched (
            article_id, model_version, language,
            sentiment_label, sentiment_score,
            dominant_topic, cpu_processing_time, gpu_processing_time
        ) VALUES (
            :aid, :mv, :lang,
            :s_lbl, :s_sc,
            :topic, :ctime, :gtime
        )
        ON DUPLICATE KEY UPDATE
            language = COALESCE(VALUES(language), language),
            sentiment_label = COALESCE(VALUES(sentiment_label), sentiment_label),
            sentiment_score = COALESCE(VALUES(sentiment_score), sentiment_score),
            dominant_topic = COALESCE(VALUES(dominant_topic), dominant_topic),
            cpu_processing_time = COALESCE(VALUES(cpu_processing_time), cpu_processing_time),
            gpu_processing_time = COALESCE(VALUES(gpu_processing_time), gpu_processing_time)
        """
        self.execute_query(sql, {
            "aid": int(article_id),
            "mv": int(model_version),
            "lang": language,
            "s_lbl": sentiment_label,
            "s_sc": sentiment_score,
            "topic": dominant_topic,
            "ctime": cpu_processing_time,
            "gtime": gpu_processing_time,
        })

    def find_similar_entity(self, normalized_name: str, entity_type: str, threshold: float = 0.9) -> int | None:
        """
        Looks for an existing entity with the same type and a very similar normalized name.
        Uses MySQL's string comparison or a simple exact match fallback if no fuzzy plugin.
        Since we want Levenshtein-like behavior without external plugins, we use 
        a list of recent similar and check top candidates.
        """
        # Exact match first (fast)
        sql_exact = "SELECT entity_id FROM entities WHERE entity_type = :typ AND normalized_name = :norm LIMIT 1"
        res = self.execute_query(sql_exact, {"typ": entity_type, "norm": normalized_name})
        row = res.fetchone()
        if row:
            return int(row[0])
        
        # Fuzzy fallback: If name is long enough, check for minor differences
        if len(normalized_name) < 5:
            return None

        # Simple heuristic: Check entities with same first 3 chars to limit search
        prefix = normalized_name[:3]
        sql_fuzzy = "SELECT entity_id, normalized_name FROM entities WHERE entity_type = :typ AND normalized_name LIKE :pref"
        res = self.execute_query(sql_fuzzy, {"typ": entity_type, "pref": f"{prefix}%"})
        
        from difflib import SequenceMatcher
        for row in res.fetchall():
            eid, existing_name = row
            ratio = SequenceMatcher(None, normalized_name, existing_name).ratio()
            if ratio >= threshold:
                return int(eid)
        
        return None

    def upsert_entity(self, entity_name: str, entity_type: str, normalized_name: str) -> int:
        """
        Inserts or updates an entity with similarity-based deduplication.
        """
        # 1. Look for similar/exact entity first
        existing_id = self.find_similar_entity(normalized_name, entity_type)
        if existing_id:
            return existing_id

        # 2. Insert if new
        sql = """
        INSERT INTO entities (entity_name, entity_type, normalized_name)
        VALUES (:name, :typ, :norm)
        """
        with self.engine.connect() as conn:
            conn.execute(text(sql), {"name": entity_name, "typ": entity_type, "norm": normalized_name})
            res = conn.execute(text("SELECT LAST_INSERT_ID()"))
            entity_id = int(res.scalar())
            conn.commit()
        return entity_id

    def update_entity_frequencies(self):
        """
        Calculates global frequency for each entity based on article_entities counts.
        """
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
        self.execute_query(sql)

    def upsert_article_entity(
        self,
        article_id: int,
        entity_id: int,
        model_version: int,
        confidence_score: float | None,
    ):
        sql = """
        INSERT INTO article_entities (article_id, entity_id, model_version, confidence_score)
        VALUES (:aid, :eid, :mv, :conf)
        ON DUPLICATE KEY UPDATE
            confidence_score = GREATEST(confidence_score, VALUES(confidence_score))
        """
        self.execute_query(sql, {
            "aid": int(article_id),
            "eid": int(entity_id),
            "mv": int(model_version),
            "conf": float(confidence_score) if confidence_score is not None else None,
        })

    def upsert_article_topic(
        self,
        article_id: int,
        topic_label: str,
        topic_score: float | None = None,
    ):
        sql = """
        INSERT INTO article_topics (article_id, topic_label, topic_score)
        VALUES (:aid, :lab, :sc)
        ON DUPLICATE KEY UPDATE
            topic_label = VALUES(topic_label),
            topic_score = VALUES(topic_score)
        """
        self.execute_query(sql, {
            "aid": int(article_id),
            "lab": str(topic_label),
            "sc": float(topic_score) if topic_score is not None else None,
        })

    def close(self):
        if self.engine:
            self.engine.dispose()
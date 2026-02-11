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
                echo=False
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
    # Table Management: Drop + Recreate all 7 result tables
    # ============================================================

    RESULT_TABLES = [
        "lang_detection",
        "preprocess_ner",
        "preprocess_sentiment",
        "preprocess_keywords",
        "ner_results",
        "sentiment_results",
        "keyword_results",
    ]

    def drop_result_tables(self):
        """Drop all result tables from previous run."""
        try:
            with self.engine.connect() as conn:
                for table in self.RESULT_TABLES:
                    conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                conn.commit()
            self.logger.info(f"Dropped {len(self.RESULT_TABLES)} result tables.")
        except Exception as e:
            self.logger.error(f"Error dropping tables: {e}")
            raise

    def create_result_tables(self):
        """Create all 7 result tables fresh."""
        try:
            self.logger.info("Creating result tables...")
            with self.engine.connect() as conn:

                # 1. Language Detection (1 row per article)
                conn.execute(text("""
                CREATE TABLE lang_detection (
                    article_id BIGINT PRIMARY KEY,
                    predicted_lang VARCHAR(10),
                    confidence FLOAT,
                    expected_lang VARCHAR(10),
                    is_correct BOOLEAN
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                # 2. Preprocessed text for NER (1 row per article)
                conn.execute(text("""
                CREATE TABLE preprocess_ner (
                    article_id BIGINT PRIMARY KEY,
                    text_ner LONGTEXT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                # 3. Preprocessed text for Sentiment (1 row per article)
                conn.execute(text("""
                CREATE TABLE preprocess_sentiment (
                    article_id BIGINT PRIMARY KEY,
                    text_sentiment LONGTEXT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                # 4. Preprocessed text for Keywords (1 row per article)
                conn.execute(text("""
                CREATE TABLE preprocess_keywords (
                    article_id BIGINT PRIMARY KEY,
                    text_keywords LONGTEXT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                # 5. NER Results (1 row per article, 3 model columns)
                conn.execute(text("""
                CREATE TABLE ner_results (
                    article_id BIGINT PRIMARY KEY,
                    arabert_entities LONGTEXT,
                    camel_entities LONGTEXT,
                    mbert_entities LONGTEXT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                # 6. Sentiment Results (1 row per article, 3 models x 3 columns)
                conn.execute(text("""
                CREATE TABLE sentiment_results (
                    article_id BIGINT PRIMARY KEY,
                    arabert_label VARCHAR(10),
                    arabert_score FLOAT,
                    arabert_probs JSON,
                    camel_label VARCHAR(10),
                    camel_score FLOAT,
                    camel_probs JSON,
                    mbert_label VARCHAR(10),
                    mbert_score FLOAT,
                    mbert_probs JSON
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                # 7. Keyword Results (1 row per article)
                conn.execute(text("""
                CREATE TABLE keyword_results (
                    article_id BIGINT PRIMARY KEY,
                    keywords LONGTEXT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))

                conn.commit()
            self.logger.info("All 7 result tables created successfully.")
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise

    def init_result_tables(self):
        """Drop previous results and create fresh tables. Call at start of each run."""
        self.drop_result_tables()
        self.create_result_tables()

    # ============================================================
    # Data Insert Methods
    # ============================================================

    def save_lang_detection(self, article_id, predicted_lang, confidence,
                            expected_lang, is_correct):
        sql = """
            INSERT INTO lang_detection
                (article_id, predicted_lang, confidence, expected_lang, is_correct)
            VALUES (:aid, :pred, :conf, :exp, :cor)
        """
        self.execute_query(sql, {
            "aid": article_id,
            "pred": predicted_lang,
            "conf": confidence,
            "exp": expected_lang,
            "cor": is_correct,
        })

    def save_preprocess_ner(self, article_id, text_ner):
        sql = """
            INSERT INTO preprocess_ner (article_id, text_ner)
            VALUES (:aid, :txt)
        """
        self.execute_query(sql, {"aid": article_id, "txt": text_ner})

    def save_preprocess_sentiment(self, article_id, text_sentiment):
        sql = """
            INSERT INTO preprocess_sentiment (article_id, text_sentiment)
            VALUES (:aid, :txt)
        """
        self.execute_query(sql, {"aid": article_id, "txt": text_sentiment})

    def save_preprocess_keywords(self, article_id, text_keywords):
        sql = """
            INSERT INTO preprocess_keywords (article_id, text_keywords)
            VALUES (:aid, :txt)
        """
        self.execute_query(sql, {"aid": article_id, "txt": text_keywords})

    def save_ner_results(self, article_id, arabert_entities,
                         camel_entities, mbert_entities):
        sql = """
            INSERT INTO ner_results
                (article_id, arabert_entities, camel_entities, mbert_entities)
            VALUES (:aid, :ara, :cam, :mbe)
        """
        self.execute_query(sql, {
            "aid": article_id,
            "ara": arabert_entities,
            "cam": camel_entities,
            "mbe": mbert_entities,
        })

    def save_sentiment_results(self, article_id,
                               arabert_label, arabert_score, arabert_probs,
                               camel_label, camel_score, camel_probs,
                               mbert_label, mbert_score, mbert_probs):
        sql = """
            INSERT INTO sentiment_results
                (article_id,
                 arabert_label, arabert_score, arabert_probs,
                 camel_label, camel_score, camel_probs,
                 mbert_label, mbert_score, mbert_probs)
            VALUES (:aid,
                    :a_lbl, :a_sc, :a_pr,
                    :c_lbl, :c_sc, :c_pr,
                    :m_lbl, :m_sc, :m_pr)
        """
        self.execute_query(sql, {
            "aid": article_id,
            "a_lbl": arabert_label,
            "a_sc": arabert_score,
            "a_pr": arabert_probs,
            "c_lbl": camel_label,
            "c_sc": camel_score,
            "c_pr": camel_probs,
            "m_lbl": mbert_label,
            "m_sc": mbert_score,
            "m_pr": mbert_probs,
        })

    def save_keyword_results(self, article_id, keywords):
        sql = """
            INSERT INTO keyword_results (article_id, keywords)
            VALUES (:aid, :kw)
        """
        self.execute_query(sql, {"aid": article_id, "kw": keywords})

    # ============================================================
    # Cleanup
    # ============================================================

    def close(self):
        if self.engine:
            self.engine.dispose()
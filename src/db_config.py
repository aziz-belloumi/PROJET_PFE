# Connection & Logging: Connects to MySQL and sends all logs to your specific Experiment folder (not the console).
# Storage: Creates the results tables (nlp_sentiment, nlp_entities, nlp_keywords, nlp_language_detection) and saves the output of your models there row-by-row.


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

    def create_nlp_tables(self): #Infrastructure: Defines the schema for NLP results.
        try:
            self.logger.info("Initializing NLP results tables...")
            with self.engine.connect() as conn:
                
                # 1. Language Detection
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nlp_language_detection (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    article_id INT NOT NULL,
                    detected_language VARCHAR(10),
                    confidence_score FLOAT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_article (article_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))
                
                # 2. Keywords
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nlp_keywords (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    article_id INT NOT NULL,
                    keyword VARCHAR(255),
                    score FLOAT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_article (article_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))
                
                # 3. Entities (NER)
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nlp_entities (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    article_id INT NOT NULL,
                    entity_text VARCHAR(255),
                    entity_type VARCHAR(50),
                    confidence_score FLOAT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_article (article_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))
                
                # 4. Sentiment
                conn.execute(text("""
                CREATE TABLE IF NOT EXISTS nlp_sentiment (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    article_id INT NOT NULL,
                    sentiment_label VARCHAR(20),
                    confidence_score FLOAT,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_article (article_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """))
                conn.commit()
                self.logger.info("✓ NLP tables are ready.")
        except Exception as e:
            self.logger.error(f"Error creating NLP tables: {e}")
            raise

    # --- Data Access Methods (Called by Pipeline)

    def save_language_detection(self, article_id, language, confidence):
        sql = "INSERT INTO nlp_language_detection (article_id, detected_language, confidence_score) VALUES (:id, :lang, :conf)"
        self.execute_query(sql, {'id': article_id, 'lang': language, 'conf': confidence})

    def save_sentiment(self, article_id, label, confidence):
        sql = "INSERT INTO nlp_sentiment (article_id, sentiment_label, confidence_score) VALUES (:id, :lbl, :conf)"
        self.execute_query(sql, {'id': article_id, 'lbl': label, 'conf': confidence})

    def save_keywords(self, article_id, keywords_list):
        # keywords_list = [(word, score), (word, score)]
        if not keywords_list: return
        sql = "INSERT INTO nlp_keywords (article_id, keyword, score) VALUES (:id, :kw, :sc)"
        with self.engine.connect() as conn:
            for kw, score in keywords_list:
                conn.execute(text(sql), {'id': article_id, 'kw': kw, 'sc': float(score)})
            conn.commit()

    def save_entities(self, article_id, entities_list):
        if not entities_list: return
        sql = "INSERT INTO nlp_entities (article_id, entity_text, entity_type, confidence_score) VALUES (:id, :txt, :typ, :conf)"
        with self.engine.connect() as conn:
            for ent in entities_list:
                conn.execute(text(sql), {
                    'id': article_id, 
                    'txt': ent['text'], 
                    'typ': ent['type'], 
                    'conf': float(ent['score'])
                })
            conn.commit()

    def close(self):
        if self.engine:
            self.engine.dispose()
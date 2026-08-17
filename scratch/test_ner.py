from src.config import Config
from src.db_config import DatabaseConnection
from src.ner_extraction import GLiNERNER
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = DatabaseConnection()

try:
    ner = GLiNERNER(model_name="urchade/gliner_multi-v2.1", logger=logger, device=0)
    print("Model loaded")
    ents = ner.predict("This is a test sentence in Arabic: محمد ذهب إلى المدرسة.", language="ar")
    print(ents)
    print("SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()

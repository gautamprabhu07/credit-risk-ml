import os


MONGO_DATABASE_NAME = os.getenv("MONGO_DATABASE_NAME", "credit_card_database")
MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "credit_card_data")


TARGET_COLUMN = "default_payment_next_month"
MONGO_DB_URL = os.getenv("MONGO_DB_URL", "")

MODEL_FILE_NAME = "model"
MODEL_FILE_EXTENSION = ".pkl"

artifact_folder = "artifacts"
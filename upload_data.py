from pymongo import MongoClient, UpdateOne
import pandas as pd
import json
import os
from src.constant import MONGO_DB_URL, MONGO_DATABASE_NAME, MONGO_COLLECTION_NAME

def upload_data() -> None:
	if not MONGO_DB_URL:
		raise ValueError("MONGO_DB_URL is not set. Please configure it as an environment variable.")

	dataset_file_path = os.path.join(os.getcwd(), "notebooks", "datasets", "UCI_Credit_Card.csv")

	if not os.path.exists(dataset_file_path):
		raise FileNotFoundError(f"Dataset not found at: {dataset_file_path}")

	client = MongoClient(MONGO_DB_URL)
	collection = client[MONGO_DATABASE_NAME][MONGO_COLLECTION_NAME]

	df = pd.read_csv(dataset_file_path)
	json_records = list(json.loads(df.T.to_json()).values())

	if not json_records:
		print("No rows found in dataset. Nothing uploaded.")
		return

	if "ID" in df.columns:
		collection.create_index("ID", unique=True)
		operations = [
			UpdateOne({"ID": record["ID"]}, {"$set": record}, upsert=True)
			for record in json_records
		]
		result = collection.bulk_write(operations, ordered=False)
		print(
			f"Upload complete. matched={result.matched_count}, modified={result.modified_count}, "
			f"upserted={len(result.upserted_ids)}"
		)
	else:
		collection.delete_many({})
		result = collection.insert_many(json_records)
		print(f"Upload complete. inserted={len(result.inserted_ids)}")


if __name__ == "__main__":
	upload_data()
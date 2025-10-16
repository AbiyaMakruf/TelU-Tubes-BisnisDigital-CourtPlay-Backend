import json
import os
from google.cloud import pubsub_v1
from supabase_utils import *
from dotenv import load_dotenv

# Project dan Topic ID Google Cloud
load_dotenv()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
TOPIC_ID = os.getenv("TOPIC_ID")

# Inisialisasi Publisher Client
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path(GCP_PROJECT_ID, TOPIC_ID)

# Fungsi untuk mengirim pesan ke Pub/Sub
def publish_message(data: dict):
    try:
        json_string = json.dumps(data)
        data_bytes = json_string.encode("utf-8")
        future = publisher.publish(topic_path, data_bytes)
        message_id = future.result() 

        print(f"[SUCCESS] Pesan untuk user_id '{data.get('user_id', 'N/A')}' berhasil dikirim. Message ID: {message_id}")

    except Exception as e:
        print(f"[ERROR] Terjadi kesalahan saat mengirim pesan untuk user_id '{data.get('user_id', 'N/A')}': {e}")

# Looping pengiriman data berdasarkan project di Supabase yang status is_mailed = False
for user_data in get("projects","id, user_id, project_details_id", {"is_mailed": False}):
    publish_message(user_data)
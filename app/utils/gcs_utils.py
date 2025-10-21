import os
import logging
import json
import google.cloud.logging
from google.cloud import storage
from google.cloud import pubsub_v1

# Inisialisasi klien GCS
storage_client = storage.Client()

# Konfigurasi Log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def download(bucket_name, gcs_source_path, local_destination_path):
    """
    Fungsi download umum dari GCS ke sistem file lokal.
    
    :returns: Path file lokal jika berhasil, None jika gagal.
    """
    
    try: 
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_source_path)
        local_dir = os.path.dirname(local_destination_path)
        os.makedirs(local_dir, exist_ok=True)
        blob.download_to_filename(local_destination_path)

        logger.info(f'Succes Download File')

    except Exception as e:
        logger.error(f'Error Download File. {e}')
    

def upload(bucket_name, local_source_path, gcs_destination_path):
    """
    Fungsi upload umum dari sistem file lokal ke GCS.
    
    :returns: URL publik file jika berhasil dan dibuat publik, atau None jika gagal.
    """

    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(gcs_destination_path)
        blob.upload_from_filename(local_source_path)
        blob.make_public()
        public_url = blob.public_url

        logger.info(f'Success Upload File')
        return public_url
    except Exception as e:
        logger.error(f'Error Upload File. {e}')
        return None
import os
from google.cloud import storage
from urllib.parse import urlparse, unquote

# Inisialisasi klien GCS
storage_client = storage.Client()

def download_model(bucket_name, model_type):
    try:
        bucket =  storage_client.bucket(bucket_name)
        blob = bucket.blob(f'assets/models/{model_type}/{model_type}.pt')

        destination_dir = f'models/{model_type}'
        destination_file = f'{destination_dir}/{model_type}.pt'
        os.makedirs(destination_dir, exist_ok=True)

        blob.download_to_filename(destination_file)

        print(f'Model {model_type} downloaded to {destination_file}')
    except Exception as e:
        print(f'Error downloading model: {e}')

def download_original_video(bucket_name, user_id, project_id, link_video):
    try:
        parsed_url = urlparse(link_video)
        path_segments = parsed_url.path.strip('/').split('/', 1)

        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(path_segments[1])

        destination_dir = f'inference/{user_id}/{project_id}'
        destination_file = f'{destination_dir}/original_video.mp4'
        os.makedirs(destination_dir, exist_ok=True)

        blob.download_to_filename(destination_file)
        print(f'Original video downloaded to {destination_file}')

    except Exception as e:
        print(f'Error downloading original video: {e}')

def upload_video(bucket_name, video_type, user_id, project_id, path_video):
    try:
        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(f'uploads/videos/{user_id}/{project_id}/{video_type}.mp4')

        blob.upload_from_filename(path_video)
        blob.make_public()
        public_url = blob.public_url
        print(f'Video uploaded to {public_url}')
        return public_url
    
    except Exception as e:
        print(f'Error uploading video: {e}')
        return None
    
def upload_thumbnail(bucket_name, user_id, project_id, path_thumbnail):
    try:
        bucket = storage_client.bucket(bucket_name)

        blob = bucket.blob(f'uploads/videos/{user_id}/{project_id}/thumbnail.jpg')

        blob.upload_from_filename(path_thumbnail)
        blob.make_public()
        public_url = blob.public_url
        print(f'Thumbnail uploaded to {public_url}')
        return public_url
    except Exception as e:
        print(f'Error uploading thumbnail: {e}')
        return None
    
# upload_video("courtplay-storage", 'object_detection', 'a01760a7-6ca7-458a-88ec-369dfd0e8154', 'a017675f-ab61-4e2d-b105-25dc10684cb3', 'inference/a01760a7-6ca7-458a-88ec-369dfd0e8154/a017675f-ab61-4e2d-b105-25dc10684cb3/original_video.mp4')
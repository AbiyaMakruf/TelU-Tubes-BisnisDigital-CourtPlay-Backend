import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_link_original_video(project_details_id):
    try:
        response = (
            supabase.table("project_details")
            .select("link_original_video")
            .eq("id", project_details_id)
            .single()
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error getting project {project_details_id}: {e}")
        return None

def get_user_info(user_id):
    try:
        response = (
            supabase.table("users")
            .select("first_name, email")
            .eq("id", user_id)
            .single()
            .execute()
        )
        return response.data['first_name'], response.data['email']
    except Exception as e:
        print(f"Error getting user {user_id}: {e}")
        return None, None
    
def get_project_info(project_id):
    try:
        response = (
            supabase.table("projects")
            .select("project_name, project_details_id")
            .eq("id", project_id)
            .single()
            .execute()
        )
        return response.data['project_name'], response.data['project_details_id']
    except Exception as e:
        print(f"Error getting project {project_id}: {e}")
        return None, None

def update_project_details(project_details_id, video_objectDetection, video_playerKeyPoint, images_ball_dropping, forehand_count, backhand_count, serve_count, video_duration, video_processing_time):
    try:
        response = (
            supabase.table("project_details")
            .update({"link_video_object_detection": video_objectDetection, 
                     "link_video_keypoints": video_playerKeyPoint, 
                     "link_images_ball_droppings": images_ball_dropping, 
                     "forehand_count": forehand_count, 
                     "backhand_count": backhand_count, 
                     "serve_count": serve_count, 
                     "video_duration": video_duration, 
                     "video_processing_time": video_processing_time, 
                     "updated_at": "now()"})
            .eq("id", project_details_id)
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error updating project {project_details_id}: {e}")
        return None

def update_projects(project_id, thumbnail, status):
    try:
        response = (
            supabase.table("projects")
            .update({"thumbnail": thumbnail,"is_mailed": status, "updated_at": "now()"})
            .eq("id", project_id)
            .execute()
        )
        return response.data
    except Exception as e:
        print(f"Error updating project {project_id}: {e}")
        return None

# print(get_link_original_video("a017675f-9297-41b2-815c-618c3fcf8b73")['link_original_video'])
# print(update("a017675f-9297-41b2-815c-618c3fcf8b73", "link_video_objectDetection", "link_video_playerKeyPoint", "images_ball_dropping", 1, 2, 3, 4, 5))
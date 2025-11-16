import os
import mailtrap as mt
import logging
import google.cloud.logging
from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader

# Mailtrap Token
load_dotenv()
MAILTRAP_TOKEN = os.getenv("MAILTRAP_TOKEN")

# Konfigurasi Log
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Konfigurasi Jinja
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
template_path = os.path.join(BASE_DIR, "email_templates")
file_loader = FileSystemLoader(template_path)
jinja_env = Environment(loader=file_loader)

# Konfigurasi Mailtrap
MAILTRAP_TOKEN = os.getenv("MAILTRAP_TOKEN")
sender_email = "support@courtplay.my.id"
sender_name = "CourtPlay Team"

def render_template(template_name, context):
    try:
        template = jinja_env.get_template(template_name)
        return template.render(context)
    except Exception as e:
        logger.error(f'Error rendering HTML email template. {e}', exc_info=True)
        raise

def send_success_analysis_video(receiver_email, context):
    """
    Fungsi umum untuk mengirim email menggunakan Mailtrap.

    context should contain:
    {
        username,
        project_name,
        video_duration,
        upload_date,
        report_url,
        heatmap_player_image_url (optional),
        heatmap_player_description (optional),
        ball_drop_image_url (optional),
        ball_drop_description (optional)
    }
    """
    try:
        html_content = render_template("success_analysis_video.html",context)

        mail = mt.Mail(
            sender=mt.Address(email=sender_email, name=sender_name),
            to=[mt.Address(email=receiver_email)],
            subject="Your Analysis is Ready!",
            html=html_content,
            category="success_analysis_video"
        )

        client = mt.MailtrapClient(token=MAILTRAP_TOKEN)
        response = client.send(mail)
        logger.info(f"Email berhasil dikirim ke {receiver_email}.")
        return response
    
    except Exception as e:
        logger.error(f"Gagal mengirim email ke {receiver_email}: {e}")
        return None
    

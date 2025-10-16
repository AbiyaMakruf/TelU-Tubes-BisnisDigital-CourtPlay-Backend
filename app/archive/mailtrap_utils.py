import os
import mailtrap as mt
from dotenv import load_dotenv

load_dotenv()

# Ambil token dari variabel lingkungan (direkomendasikan)
MAILTRAP_TOKEN = os.getenv("MAILTRAP_TOKEN") 

def send_success_email(
    username: str, 
    project_name: str,
    project_id: str,
    receiver_email: str,
    video_duration: int
):
    
    # 1. Definisikan Template HTML (di sini disingkat, Anda harus menggunakan template lengkap)
    # NOTE: Gantilah konten variabel di dalam string ini ({...}) dengan data dari argumen fungsi.
    
    # PENTING: Karena HTML yang Anda berikan adalah MIME-Encoded, 
    # Anda harus menggunakan versi HTML murni sebelum di-encode. 
    # Saya akan mengasumsikan Anda memiliki URL dasar untuk laporan.
    
    REPORT_URL = f"https://courtplay.my.id/analysis/{project_id}" 
    
    HTML_TEMPLATE = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Analysis Complete - CourtPlay</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600;700&display=swap');

        body {{
            background-color: #1c1c1c; /* black-200 */
            color: #fafafa; /* white-500 */
            font-family: 'Lexend', sans-serif;
            margin: 0;
            padding: 0;
            line-height: 1.6;
        }}

        .container {{
            background-color: #292929; /* black-300 */
            border-radius: 16px;
            padding: 40px;
            max-width: 600px;
            margin: 40px auto;
        }}

        .logo {{
            display: block;
            margin: 0 auto 24px;
            width: 140px;
        }}

        h2 {{
            color: #f4fdca; /* primary-500 */
            font-weight: 700;
            margin-bottom: 8px;
        }}

        p {{
            color: #d4d4d4; /* white-200 */
            margin-bottom: 15px;
        }}

        .highlight {{
            color: #a3ce14; /* primary-300 */
            font-weight: 600;
        }}

        .button {{
            display: inline-block;
            background-color: #a3ce14; /* primary-300 */
            color: #1c1c1c; /* black-200 */
            text-decoration: none;
            padding: 12px 25px;
            border-radius: 8px;
            font-weight: 700;
            margin-top: 20px;
            text-align: center;
        }}

        .analysis-summary {{
            background-color: #1c1c1c; /* black-200 */
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            font-size: 0.95em;
        }}
        .analysis-summary p {{
            margin: 5px 0;
            color: #fafafa;
        }}
        .analysis-summary span {{
            font-weight: 600;
            color: #f4fdca; /* primary-500 */
        }}

        .footer {{
            color: #888;
            font-size: 0.85em;
            margin-top: 40px;
            border-top: 1px solid #444;
            padding-top: 20px;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <img src="https://storage.googleapis.com/courtplay-storage/assets/Web/Logo-Horizontal.png" alt="CourtPlay Logo" class="logo">

        <h2>Hello, {username}! 🎉</h2>

        <p>Your video analysis project is complete! The CourtPlay AI has processed **{project_name}** and your full report is now ready.</p>

        <div class="analysis-summary">
            <p><span class="highlight">Project Name:</span> {project_name}</p>
            <p style="margin-top: 10px;"><span>Video Duration:</span> {video_duration} seconds</p>
        </div>

        <p>Click the button below to view your game metrics, shot heatmaps, and all performance insights. It's time to see your progress!</p>

        <a href="{REPORT_URL}" class="button">
            View Analysis Report 📈
        </a>

        <p style="margin-top: 30px;">Happy analyzing and take your game to the next level! 🚀</p>

        <div class="footer">
            <p>This email was sent automatically. If you have any questions, please reply to this email or visit our Help Center.<br>
            &copy; 2025 CourtPlay. All rights reserved.</p>
        </div>
    </div>
</body>
</html>
"""
    
    # 2. Buat objek Mailtrap
    mail = mt.Mail(
        sender=mt.Address(email="support@courtplay.my.id", name="CourtPlay Team"),
        to=[mt.Address(email=receiver_email)],
        subject=f"Your Analysis is Ready! | CourtPlay",
        html=HTML_TEMPLATE,
        category="analysis_complete",
    )

    # 3. Kirim Email
    client = mt.MailtrapClient(token=MAILTRAP_TOKEN)
    response = client.send(mail)
    return response
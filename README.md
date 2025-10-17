![CourtPlay Logo](README/CP_BE.png)

![Tech Stack](README/tech_stack.png)

# Build Docker Local
docker build -t us-central1-docker.pkg.dev/courtplay-analytics-474615/courtplay-repo/backend-app:latest .

# Run Docker Local
docker run -it ^
  -p 8080:8080 ^
  -v "C:\Users\abiyamf\Documents\Bisnis Digital\TelU-Tubes-BisnisDigital-CourtPlay-Backend\courtplay-analytics-474615-92b2bbfdc547.json:/app/gcp-key.json" ^
  -e GOOGLE_APPLICATION_CREDENTIALS="/app/gcp-key.json" ^
  -e SUPABASE_URL="https://qqcxjwxphvqfiffwftpt.supabase.co" ^
  -e SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFxY3hqd3hwaHZxZmlmZndmdHB0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjAyMDg0NDAsImV4cCI6MjA3NTc4NDQ0MH0.ileKiwiqCwpBT0FGRxoyd9pAKrrluiH1xGjpsRxas5w" ^
  -e MAILTRAP_TOKEN="ce9a293bbce7cd6e2140ec38f3c6a323" ^
  -e GCP_PROJECT_ID="courtplay-analytics-474615" ^
  -e TOPIC_ID="courtplay-pubSub" ^
  -e SUBSCRIPTION_ID="inference" ^
  -e MAX_CONCURRENT_MESSAGES=1 ^
  us-central1-docker.pkg.dev/courtplay-analytics-474615/courtplay-repo/backend-app:latest

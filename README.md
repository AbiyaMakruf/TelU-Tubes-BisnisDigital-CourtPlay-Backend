![CourtPlay Logo](README/CP_BE.png)

![Tech Stack](README/tech_stack.png)

# Build Docker Local
docker build -t us-central1-docker.pkg.dev/courtplay-analytics-474615/courtplay-repo/backend-app:latest .

uvicorn app.main:app --reload
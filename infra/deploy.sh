#!/usr/bin/env bash
#
# Deploy Orly backend to Cloud Run.
#
# Required env vars:
#   GCP_PROJECT_ID — Google Cloud project ID
#
# Optional env vars:
#   GCP_REGION — deployment region (default: us-central1)
#   SERVICE_NAME — Cloud Run service name (default: orly-backend)

set -euo pipefail

# --- Validate required env vars ---
if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
    echo "ERROR: GCP_PROJECT_ID environment variable is required."
    echo "Usage: GCP_PROJECT_ID=orly-490422 ./infra/deploy.sh"
    exit 1
fi

REGION="${GCP_REGION:-us-central1}"
SERVICE="${SERVICE_NAME:-orly-backend}"

echo "=== Orly Backend Deployment ==="
echo "Project:  ${GCP_PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE}"
echo ""

# --- Set active project ---
gcloud config set project "${GCP_PROJECT_ID}"

# --- Enable required APIs ---
echo "Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com \
    artifactregistry.googleapis.com

# --- Deploy using Cloud Build (no local Docker required) ---
# gcloud run deploy --source uses the Dockerfile at the repo root,
# builds it remotely via Cloud Build, pushes to Artifact Registry,
# and deploys to Cloud Run in one step.
echo "Building and deploying via Cloud Build..."
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

gcloud run deploy "${SERVICE}" \
    --source . \
    --region "${REGION}" \
    --platform managed \
    --session-affinity \
    --min-instances=0 \
    --timeout=3600 \
    --allow-unauthenticated \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${GCP_PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}"

# --- Print deployed URL ---
echo ""
echo "=== Deployment complete ==="
URL=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format "value(status.url)")
echo "Service URL: ${URL}"
echo "WebSocket:   ${URL/https/wss}/ws/session"

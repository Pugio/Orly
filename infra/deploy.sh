#!/usr/bin/env bash
#
# Deploy TableLight backend to Cloud Run.
#
# Required env vars:
#   GCP_PROJECT_ID — Google Cloud project ID
#
# Optional env vars:
#   GCP_REGION — deployment region (default: us-central1)
#   SERVICE_NAME — Cloud Run service name (default: tablelight-backend)

set -euo pipefail

# --- Validate required env vars ---
if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
    echo "ERROR: GCP_PROJECT_ID environment variable is required."
    echo "Usage: GCP_PROJECT_ID=my-project ./infra/deploy.sh"
    exit 1
fi

REGION="${GCP_REGION:-us-central1}"
SERVICE="${SERVICE_NAME:-tablelight-backend}"
IMAGE="gcr.io/${GCP_PROJECT_ID}/${SERVICE}"

echo "=== TableLight Backend Deployment ==="
echo "Project:  ${GCP_PROJECT_ID}"
echo "Region:   ${REGION}"
echo "Service:  ${SERVICE}"
echo "Image:    ${IMAGE}"
echo ""

# --- Set active project ---
gcloud config set project "${GCP_PROJECT_ID}"

# --- Enable required APIs ---
echo "Enabling required APIs..."
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    aiplatform.googleapis.com

# --- Build the Docker image ---
echo "Building Docker image..."
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
docker build -t "${IMAGE}" -f "${REPO_ROOT}/infra/Dockerfile" "${REPO_ROOT}"

# --- Push to Container Registry ---
echo "Pushing image to GCR..."
docker push "${IMAGE}"

# --- Deploy to Cloud Run ---
echo "Deploying to Cloud Run..."
gcloud run deploy "${SERVICE}" \
    --image "${IMAGE}" \
    --region "${REGION}" \
    --platform managed \
    --session-affinity \
    --min-instances=1 \
    --timeout=3600 \
    --allow-unauthenticated \
    --set-env-vars "GOOGLE_CLOUD_PROJECT=${GCP_PROJECT_ID},GOOGLE_CLOUD_LOCATION=${REGION}"

# --- Print deployed URL ---
echo ""
echo "=== Deployment complete ==="
URL=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format "value(status.url)")
echo "Service URL: ${URL}"
echo "WebSocket:   ${URL/https/wss}/ws/session"

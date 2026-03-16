FROM python:3.12-slim

WORKDIR /app

# Install only backend production dependencies (client-side packages excluded)
RUN pip install --no-cache-dir \
    google-genai \
    fastapi \
    "uvicorn[standard]" \
    numpy \
    websockets

# Copy only the backend module
COPY backend/ backend/

EXPOSE 8080

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]

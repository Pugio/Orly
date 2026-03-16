"""Tests to verify Cloud Run deployment artifacts exist and have expected content."""

from pathlib import Path

INFRA_DIR = Path(__file__).resolve().parent.parent / "infra"


def test_dockerfile_exists():
    dockerfile = INFRA_DIR / "Dockerfile"
    assert dockerfile.exists(), "infra/Dockerfile must exist"


def test_dockerfile_has_base_image():
    content = (INFRA_DIR / "Dockerfile").read_text()
    assert "FROM python:3.12-slim" in content


def test_dockerfile_copies_backend():
    content = (INFRA_DIR / "Dockerfile").read_text()
    assert "COPY backend/" in content


def test_dockerfile_has_cmd():
    content = (INFRA_DIR / "Dockerfile").read_text()
    assert "CMD" in content
    assert "uvicorn" in content
    assert "backend.main:app" in content


def test_dockerfile_installs_google_genai():
    content = (INFRA_DIR / "Dockerfile").read_text()
    assert "google-genai" in content


def test_dockerfile_does_not_include_client_deps():
    content = (INFRA_DIR / "Dockerfile").read_text()
    assert "pyaudio" not in content
    assert "opencv" not in content.lower()


def test_deploy_script_exists_and_executable():
    deploy = INFRA_DIR / "deploy.sh"
    assert deploy.exists(), "infra/deploy.sh must exist"
    import os

    assert os.access(deploy, os.X_OK), "deploy.sh must be executable"


def test_deploy_script_has_session_affinity():
    content = (INFRA_DIR / "deploy.sh").read_text()
    assert "--session-affinity" in content


def test_deploy_script_has_min_instances():
    content = (INFRA_DIR / "deploy.sh").read_text()
    assert "--min-instances=" in content


def test_deploy_script_has_timeout():
    content = (INFRA_DIR / "deploy.sh").read_text()
    assert "--timeout=3600" in content


def test_deploy_script_requires_project_id():
    content = (INFRA_DIR / "deploy.sh").read_text()
    assert "GCP_PROJECT_ID" in content


def test_dockerignore_exists():
    dockerignore = INFRA_DIR / ".dockerignore"
    assert dockerignore.exists(), "infra/.dockerignore must exist"


def test_dockerignore_excludes_client():
    content = (INFRA_DIR / ".dockerignore").read_text()
    assert "client/" in content
    assert "tests/" in content
    assert ".env" in content


def test_cloudbuild_yaml_exists():
    cloudbuild = INFRA_DIR / "cloudbuild.yaml"
    assert cloudbuild.exists(), "infra/cloudbuild.yaml must exist"

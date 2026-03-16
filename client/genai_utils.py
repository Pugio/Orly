"""Shared google-genai client singleton for edge-side modules.

Used by image generation, video generation, code generation, and music.
"""

import os


_genai_client_cache = {}


def get_genai_client():
    """Get or create a cached google.genai client.

    Checks env vars first, then falls back to ``llm keys get gemini``.
    """
    if "client" in _genai_client_cache:
        return _genai_client_cache["client"]
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        import subprocess

        try:
            api_key = subprocess.check_output(
                ["llm", "keys", "get", "gemini"], text=True
            ).strip()
        except Exception:
            pass
    if not api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GOOGLE_API_KEY or run `llm keys set gemini`."
        )
    client = genai.Client(api_key=api_key)
    _genai_client_cache["client"] = client
    return client

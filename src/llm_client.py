import os
from typing import Optional

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def get_openai_client() -> Optional["OpenAI"]:
    """Return a configured OpenAI client, or None if the SDK or API key is unavailable."""
    if OpenAI is None:
        return None
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

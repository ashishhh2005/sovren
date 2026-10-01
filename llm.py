"""
Shared LLM client. Sovren uses Google Gemini through its OpenAI-compatible API,
so we keep using the familiar `openai` SDK — only the base URL, key, and model
names change. Gemini has a free tier (no credit card), which is why it's used here.
"""

import os

# Gemini exposes an OpenAI-compatible endpoint, so the openai SDK works unchanged.
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
CHAT_MODEL = "gemini-2.0-flash"
EMBED_MODEL = "gemini-embedding-001"


def get_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


def client():
    """Return an OpenAI-SDK client pointed at Gemini. Caller checks get_key() first."""
    from openai import OpenAI

    return OpenAI(api_key=get_key(), base_url=GEMINI_BASE)

import os
import gradio as gr
from google import genai
from google.genai import types

BUILD_MARKER = "MODULE3_CHAT_APP_STABLE_V2"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
MAX_HISTORY = 12
MAX_CHARS = 2000


def make_contents(history, current_message):
    """Convert Gradio message history into a simple Gemini conversation prompt."""
    parts = []
    for item in (history or [])[-MAX_HISTORY:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            label = "User" if role == "user" else "Assistant"
            parts.append(f"{label}: {content[:MAX_CHARS]}")
    parts.append(f"User: {current_message[:MAX_CHARS]}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def chat(message, history, instructions):
    message = (message or "").strip()
    history = history or []
    if not message:
        return history, "", "Please type a message first."

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        answer = "The app owner must add GEMINI_API_KEY in Render → Environment."
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ], "", "Gemini API key is not configured."

    try:
        system_instruction = (instructions or "").strip()
        if not system_instruction:
            system_instruction = "You are a helpful assistant. Explain things clearly and simply."

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=make_contents(history, message),
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
                max_output_tokens=512,
            ),
        )
        answer = (getattr(response, "text", None) or "").strip()
        if not answer:
            answer = "Gemini returned an empty response. Please try again."

        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            usage_text = "Token usage information was not returned for this request."
        else:
            input_tokens = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)
            usage_text = (
                "**Token usage (this request):** "
                f"input `{input_tokens if input_tokens is not None else 'n/a'}`, "
                f"output `{output_tokens if output_tokens is not None else 'n/a'}`, "
                f"total `{total_tokens if total_tokens is not None else 'n/a'}`."
            )
    except Exception as exc:
        # Do not print or expose the API key.
        print(f"{BUILD_MARKER} request error: {type(exc).__name__}: {exc}", flush=True)
        details = str(exc).upper()
        if "API_KEY_INVALID" in details or "UNAUTHENTICATED" in details or "401" in details:
            answer = "Authentication failed. Check the GEMINI_API_KEY value in Render."
        elif "PERMISSION_DENIED" in details or "403" in details:
            answer = "Gemini denied access. Check that the API key has Gemini API access."
        elif "RESOURCE_EXHAUSTED" in details or "429" in details or "QUOTA" in details:
            answer = "The Gemini API quota or rate limit was reached. Try again later or check your quota."
        elif "NOT_FOUND" in details or "404" in details or "MODEL" in details and "NOT FOUND" in details:
            answer = (
                f"The model '{MODEL_NAME}' is unavailable for this API key. "
                "Set GEMINI_MODEL in Render to the model ID that worked in your Colab notebook."
            )
        else:
            answer = f"Gemini request failed ({type(exc).__name__}). Check the latest Render logs."
        usage_text = "Token usage is available after a successful response."

    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

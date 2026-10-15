import os
import gradio as gr
from google import genai
from google.genai import types

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def create_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in the Render environment.")
    return genai.Client(api_key=api_key)


def chat(message, history, instructions):
    history = history or []
    message = (message or "").strip()
    instructions = (instructions or "").strip()

    if not message:
        return history, "", "Please enter a message."

    contents = []
    # Include recent conversation turns so Gemini has context.
    for item in history[-12:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            contents.append(
                types.Content(
                    role="user" if role == "user" else "model",
                    parts=[types.Part.from_text(text=content)],
                )
            )

    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=message)])
    )

    system_instruction = instructions or "Answer helpfully, accurately, and clearly."

    try:
        client = create_client()
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,
                max_output_tokens=512,
            ),
        )

        answer = (response.text or "").strip()
        if not answer:
            answer = "Gemini returned an empty response. Please try again."

        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]

        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            usage_text = "Token usage metadata was not returned."
        else:
            input_count = getattr(usage, "prompt_token_count", None)
            output_count = getattr(usage, "candidates_token_count", None)
            total_count = getattr(usage, "total_token_count", None)
            usage_text = (
                f"**Token usage** — Input: {input_count if input_count is not None else 'N/A'} · "
                f"Output: {output_count if output_count is not None else 'N/A'} · "
                f"Total: {total_count if total_count is not None else 'N/A'}"
            )

        return updated_history, "", usage_text

    except Exception as exc:
        # Log diagnostic information, but never log the API key.
        print(f"Gemini API error ({type(exc).__name__}): {exc}", flush=True)
        error_text = f"{type(exc).__name__}: {exc}".upper()

        if "API_KEY_INVALID" in error_text or "UNAUTHENTICATED" in error_text or "401" in error_text:
            answer = "Authentication failed. Check GEMINI_API_KEY in Render Environment, then redeploy."
        elif "PERMISSION_DENIED" in error_text or "403" in error_text:
            answer = "Permission denied. Check that your Gemini API key and API permissions are correct."
        elif "RESOURCE_EXHAUSTED" in error_text or "429" in error_text or "QUOTA" in error_text:
            answer = "Gemini API quota or rate limit reached. Check your API usage and limits."
        elif "NOT_FOUND" in error_text or "404" in error_text:
            answer = f"The model '{MODEL_NAME}' was not found. Check the GEMINI_MODEL setting in Render."
        elif "503" in error_text or "UNAVAILABLE" in error_text:
            answer = "Gemini is temporarily unavailable. Please try again shortly."
        else:

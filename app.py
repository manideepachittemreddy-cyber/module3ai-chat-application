import os
import gradio as gr
from google import genai
from google.genai import types

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it in Render > Environment."
        )
    return genai.Client(api_key=api_key)


def chat(message, history, custom_instructions):
    history = history or []
    message = (message or "").strip()
    custom_instructions = (custom_instructions or "").strip()

    if not message:
        return history, "", "Enter a message first."

    # Keep only valid, recent user/assistant messages for context.
    contents = []
    for item in history[-12:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        text = item.get("content")
        if role in ("user", "assistant") and isinstance(text, str) and text.strip():
            contents.append(
                types.Content(
                    role="user" if role == "user" else "model",
                    parts=[types.Part.from_text(text=text)],
                )
            )

    contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=message)])
    )

    system_instruction = (
        custom_instructions
        if custom_instructions
        else "Answer helpfully, accurately, and clearly."
    )

    try:
        client = get_client()
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

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]

        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)
            usage_text = (
                f"**Token usage** — Input: {input_tokens if input_tokens is not None else 'N/A'} · "
                f"Output: {output_tokens if output_tokens is not None else 'N/A'} · "
                f"Total: {total_tokens if total_tokens is not None else 'N/A'}"
            )
        else:
            usage_text = "Token usage metadata was not returned for this response."

        return history, "", usage_text

    except Exception as exc:
        # Diagnostic details go to Render logs; never print the API key.
        print(f"Gemini API error ({type(exc).__name__}): {exc}", flush=True)

        details = f"{type(exc).__name__}: {exc}"
        upper = details.upper()

        if "API KEY" in upper or "API_KEY_INVALID" in upper or "UNAUTHENTICATED" in upper or "401" in upper:
            answer = "Authentication failed. In Render > Environment, check that GEMINI_API_KEY contains a valid Gemini API key. Save and redeploy."
        elif "PERMISSION_DENIED" in upper or "403" in upper:

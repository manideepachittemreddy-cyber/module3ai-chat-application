import os
import gradio as gr
from google import genai
from google.genai import types

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_OUTPUT_TOKENS = 512
MAX_HISTORY_MESSAGES = 10
MAX_MESSAGE_CHARS = 1500


def build_prompt(history, message):
    lines = []
    for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", "")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content[:MAX_MESSAGE_CHARS]}")
    lines.append(f"User: {message[:MAX_MESSAGE_CHARS]}")
    lines.append("Assistant:")
    return "\n".join(lines)


def chat(message, history, custom_instructions):
    message = (message or "").strip()
    history = history or []
    if not message:
        return history, "", "Enter a message to start chatting."

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        answer = (
            "The app owner needs to add GEMINI_API_KEY in the hosting service's "
            "environment/secrets settings."
        )
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return updated_history, "", "API key is not configured."

    try:
        client = genai.Client(api_key=api_key)
        instructions = (custom_instructions or "").strip()
        if not instructions:
            instructions = (
                "You are a friendly assistant. "
                "Explain answers clearly in simple language."
            )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=build_prompt(history, message),
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                temperature=0.7,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        answer = response.text or "The model returned no text. Please try again."
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)
            usage_text = (
                "**Token usage (this request):** "
                f"input `{input_tokens if input_tokens is not None else 'n/a'}`, "
                f"output `{output_tokens if output_tokens is not None else 'n/a'}`, "
                f"total `{total_tokens if total_tokens is not None else 'n/a'}`. "
                "Usage is not the same as price; actual cost depends on the model "
                "and account pricing."
            )
        else:
            usage_text = "Token usage metadata was not returned for this request."

    except Exception as exc:
        error_text = str(exc).upper()
        if "503" in error_text or "UNAVAILABLE" in error_text:
            answer = "Gemini is temporarily busy. Please wait a moment and try again."
        elif "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
            answer = (
                "The API quota or rate limit was reached. Check your Gemini API "
                "usage and try again later."
            )
        elif "401" in error_text or "403" in error_text or "API_KEY_INVALID" in error_text:
            answer = (
                "Authentication failed. The app owner should check the "
                "GEMINI_API_KEY secret and API permissions."
            )
        else:
            answer = (
                f"The request failed ({type(exc).__name__}). "
                "Please try again or contact the app owner."
            )
        usage_text = "Token usage is shown after successful API responses."

    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, "", usage_text


with gr.Blocks(
    title="AI Productivity Chat",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("# 🤖 AI Productivity Chat")
    gr.Markdown(
        "Ask questions, keep context across messages, and customize how Gemini responds. "
        "Please do not enter private or sensitive information."
    )

    instructions = gr.Textbox(
        label="Custom Instructions",
        value=(
            "You are a friendly assistant. Explain answers clearly in simple language "
            "with examples."
        ),
        placeholder="Example: Answer in two sentences and use simple English.",
        lines=2,
    )

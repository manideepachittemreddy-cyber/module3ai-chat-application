import os
import gradio as gr
from google import genai
from google.genai import types

BUILD_MARKER = "MODULE3_CHAT_APP_GRADIO6_FIXED"
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
MAX_HISTORY = 12
MAX_CHARS = 2000


def make_contents(history, current_message):
    """Build a bounded conversation prompt from Gradio's messages history."""
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
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return updated_history, "", "Gemini API key is not configured."

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
        # Never log or expose the API key.
        print(f"{BUILD_MARKER} request error: {type(exc).__name__}: {exc}", flush=True)
        details = str(exc).upper()
        if "API_KEY_INVALID" in details or "UNAUTHENTICATED" in details or "401" in details:
            answer = "Authentication failed. Check GEMINI_API_KEY in Render Environment settings."
        elif "PERMISSION_DENIED" in details or "403" in details:
            answer = "Gemini denied access. Check that your API key has Gemini API access."
        elif "RESOURCE_EXHAUSTED" in details or "429" in details or "QUOTA" in details:
            answer = "The Gemini API quota or rate limit was reached. Check your quota and try again."
        elif "NOT_FOUND" in details or "404" in details or ("MODEL" in details and "NOT FOUND" in details):
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
    return updated_history, "", usage_text


# Gradio 6 uses messages-format chat history by default; the old `type=`
# constructor argument is not accepted by the version currently on Render.
with gr.Blocks(title="AI Productivity Chat") as demo:
    gr.Markdown("# AI Productivity Chat\nChat with Gemini and review token usage.")
    chatbot = gr.Chatbot(label="Conversation", height=420)
    instructions = gr.Textbox(
        label="Custom instructions (optional)",
        placeholder="For example: Explain answers simply with examples.",
        lines=2,
    )
    usage_display = gr.Markdown("Token usage will appear here.")
    message = gr.Textbox(
        label="Your message",
        placeholder="Type a message and press Enter...",
        lines=2,
    )
    with gr.Row():
        send_button = gr.Button("Send", variant="primary")
        clear_button = gr.Button("Clear conversation")

    send_button.click(
        fn=chat,
        inputs=[message, chatbot, instructions],
        outputs=[chatbot, message, usage_display],
    )
    message.submit(
        fn=chat,
        inputs=[message, chatbot, instructions],
        outputs=[chatbot, message, usage_display],
    )
    clear_button.click(
        fn=lambda: ([], "", "Token usage will appear here."),
        inputs=[],
        outputs=[chatbot, message, usage_display],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    print(f"{BUILD_MARKER} starting; model={MODEL_NAME}; port={port}", flush=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        theme=gr.themes.Soft(),
    )

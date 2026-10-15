    if not message:
        return history, "", "Please enter a message."

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        answer = "The app owner must set GEMINI_API_KEY in Render Environment variables."
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer},
        ]
        return updated_history, "", "API key is not configured."

    try:
        instructions = (custom_instructions or "").strip()
        if not instructions:
            instructions = "You are a friendly assistant. Explain answers clearly in simple language."

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=make_prompt(history, message),
            config=types.GenerateContentConfig(
                system_instruction=instructions,
                temperature=0.4,
                max_output_tokens=512,
            ),
        )
        answer = (response.text or "").strip()
        if not answer:
            answer = "Gemini returned an empty response. Please try again."

        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            usage_text = "Token usage metadata was not returned."
        else:
            input_tokens = getattr(usage, "prompt_token_count", None)
            output_tokens = getattr(usage, "candidates_token_count", None)
            total_tokens = getattr(usage, "total_token_count", None)
            usage_text = (
                f"**Token usage (this request):** input `{input_tokens if input_tokens is not None else 'n/a'}`, "
                f"output `{output_tokens if output_tokens is not None else 'n/a'}`, "
                f"total `{total_tokens if total_tokens is not None else 'n/a'}`."
            )
    except Exception as exc:
        # Log the exception for debugging; never print the API key.
        print(f"{BUILD_MARKER} API error: {type(exc).__name__}: {exc}", flush=True)
        error_text = str(exc).upper()
        if "401" in error_text or "API_KEY_INVALID" in error_text or "UNAUTHENTICATED" in error_text:
            answer = "Authentication failed. Check GEMINI_API_KEY in Render."
        elif "403" in error_text or "PERMISSION_DENIED" in error_text:
            answer = "Gemini denied access. Check the API key permissions."
        elif "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "QUOTA" in error_text:
            answer = "The Gemini API quota or rate limit was reached."
        elif "404" in error_text or "NOT_FOUND" in error_text:
            answer = f"The model '{MODEL_NAME}' was not found. Check GEMINI_MODEL in Render."
        else:
            answer = f"Gemini request failed ({type(exc).__name__}). Check the Render logs."
        usage_text = "Token usage is shown after a successful response."

    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]
    return updated_history, "", usage_text


def clear_chat():
    return [], "", "**Token usage:** Send a message to see token counts."


with gr.Blocks(title="AI Productivity Chat", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# AI Productivity Chat")
    gr.Markdown("Chat with Gemini, keep recent context, and customize responses.")

    instructions = gr.Textbox(
        label="Custom Instructions",
        value="Explain answers clearly in simple English and include examples when useful.",
        lines=2,
    )
    chatbot = gr.Chatbot(label="Conversation", type="messages", height=420)
    message = gr.Textbox(
        label="Your Message",
        placeholder="Type a question and press Enter...",
        lines=2,
    )
    with gr.Row():
        send_button = gr.Button("Send", variant="primary")
        clear_button = gr.Button("Clear Chat")
    usage_display = gr.Markdown("**Token usage:** Send a message to see token counts.")

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
        fn=clear_chat,
        inputs=None,
        outputs=[chatbot, message, usage_display],
    )


if __name__ == "__main__":
    print(f"{BUILD_MARKER} starting with model {MODEL_NAME}", flush=True)
    port = int(os.environ.get("PORT", "10000"))
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

# AI Productivity Chat — Module 3

A Python chatbot web app built with Gradio and the Gemini API. It includes a chat interface, recent conversation context, custom instructions, basic error handling, and per-request token usage reporting.

## Features
- Connects to Gemini through the `google-genai` Python SDK.
- Keeps recent messages in the active chat session.
- Lets users provide custom instructions.
- Reports input, output, and total tokens when the API returns usage metadata.
- Limits output tokens and the recent history sent to the model to help manage usage.
- Handles common availability, quota, and authentication errors with user-friendly messages.

## Run locally

1. Install Python 3.10+.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set the `GEMINI_API_KEY` environment variable without putting the key in source code.
4. Run:

   ```bash
   python app.py
   ```

5. Open the local URL shown in the terminal.

## Deploy on Render

1. Upload `app.py`, `requirements.txt`, and this `README.md` to your GitHub repository.
2. In Render, create a **New Web Service** connected to that repository.
3. Use these settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python app.py`
   - **Instance type:** Free (if available on your account)
4. In the service's **Environment** settings, add a secret environment variable:
   - Key: `GEMINI_API_KEY`
   - Value: your Gemini API key
5. Optionally add `GEMINI_MODEL` with value `gemini-3.5-flash-lite` if you want to configure the model explicitly.
6. Deploy. Render will give you a public `onrender.com` URL. Test it before submitting.

Do not commit API keys, `.env` files, or secrets to GitHub. Anyone who can access a public chatbot may be able to cause API requests and use your model quota. Monitor usage and rotate the key if it is exposed.

## Testing checklist
- Ask a question and verify an AI response appears.
- Ask a follow-up question and check the chat context.
- Change custom instructions and verify the response style changes.
- Clear the conversation.
- Confirm token counts appear when supported by the API.
- Try a temporary service/quota issue if available; do not intentionally expose secret information.

## Cost and usage notes
The app limits generated output to 512 tokens and includes at most the 10 most recent history messages, with each history message truncated to 1,500 characters before being sent. These controls help limit usage but do not guarantee zero cost. Actual prices and free-tier quotas depend on the selected Gemini model and account. Review your Google AI Studio usage and billing settings.

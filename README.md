# 🖼️ Image Summary & Q&A

Chat with your images, fully offline. Upload a picture, get a detailed summary, then ask as many follow-up questions as you like — the model remembers the earlier turns of the conversation. When you're done, download the whole session as a Markdown file.

Runs on local vision models through [Ollama](https://ollama.com).

## Features

- 📝 Two-paragraph, 500+ word image summary
- 💬 Chat-style Q&A with follow-up context (the last few turns are sent with each question)
- ⬇️ Export summary + conversation as Markdown
- 🧹 Conversation resets automatically when you upload a new image
- 🛟 Clear error messages when Ollama isn't running or a model is missing

## Run it

```bash
pip install -r requirements.txt
ollama pull llava-phi3          # or moondream / qwen2.5vl
streamlit run app.py
```

## Models

| Model | Notes |
|---|---|
| `llava-phi3` | Good default — fast and reasonably accurate |
| `moondream` | Smallest and fastest |
| `qwen2.5vl` | Most detailed answers, needs more RAM |

import base64
import csv
import io
import hashlib
from datetime import datetime

import requests
import streamlit as st

OLLAMA_API = "http://localhost:11434/api/generate"

MODEL_CHOICES = ["llava-phi3", "moondream", "qwen2.5vl"]

SUMMARY_PROMPT = (
    "You are an expert vision describer. Look ONLY at the image. "
    "Write a detailed description in EXACTLY TWO PARAGRAPHS totaling AT LEAST 500 words. "
    "Be precise about objects, layout, colors, lighting, background, relationships, and possible context. "
    "Avoid speculation beyond what is visible. Do not include lists or headings; use natural prose."
)


def query_ollama(model: str, prompt: str, image_bytes: bytes, num_predict: int = 900, timeout=300):
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [base64.b64encode(image_bytes).decode("utf-8")],
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.4,
        },
    }
    resp = requests.post(OLLAMA_API, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def safe_query(model, prompt, image_bytes, num_predict):
    try:
        return query_ollama(model, prompt, image_bytes, num_predict=num_predict)
    except requests.ConnectionError:
        st.error("Can't reach Ollama at localhost:11434. Start it with `ollama serve`.")
    except requests.Timeout:
        st.error("The model timed out. Try a smaller image or fewer max tokens.")
    except requests.HTTPError as e:
        st.error(f"Ollama error — is `{model}` pulled? (`ollama pull {model}`)\n\n{e}")
    return None


LANGUAGES = ["English", "Hindi", "Kannada", "Tamil", "Telugu", "Malayalam", "Marathi",
             "Bengali", "Gujarati", "Spanish", "French", "German", "Japanese"]


def language_instruction(language: str) -> str:
    """Appended to every prompt. Small vision models drift back to English,
    so the instruction is explicit and repeated at the end of the prompt."""
    if language == "English":
        return ""
    return (f"\n\nIMPORTANT: Write your entire answer in {language}, using {language} script. "
            f"Do not answer in English.")


def build_question_prompt(question: str, history: list) -> str:
    """Include earlier Q&A turns so follow-up questions have context."""
    context = ""
    for turn in history[-4:]:
        context += f"Q: {turn['q']}\nA: {turn['a']}\n"
    return (
        "Answer clearly and concisely based ONLY on the image.\n"
        + (f"Earlier conversation about this image:\n{context}\n" if context else "")
        + f"Question: {question}"
    )


def transcript(name: str, model: str, summary, history: list, language: str = "English") -> str:
    lines = [f"# Image analysis: {name}", f"_Model: {model} · Language: {language} · {datetime.now():%Y-%m-%d %H:%M}_", ""]
    if summary:
        lines += ["## Summary", summary, ""]
    if history:
        lines.append("## Questions & answers")
        for turn in history:
            lines += [f"**Q:** {turn['q']}", "", f"**A:** {turn['a']}", ""]
    return "\n".join(lines)


st.set_page_config(page_title="Image Summary & Q&A", layout="centered")
st.title("🖼️ Image Summary & Q&A")
st.caption("Chat with your image — ask follow-up questions and download the whole session.")

with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox("Model", MODEL_CHOICES, index=0)
    language = st.selectbox("Answer language", LANGUAGES, index=0,
                            help="Summary and answers are written in this language. You can ask questions in any language.")
    max_tokens = st.slider("Max tokens (higher = longer, slower)", 200, 1400, 900, step=50)
    if language != "English":
        st.caption("Non-Latin scripts use more tokens — raise max tokens if answers get cut off. "
                   "qwen2.5vl handles Indian languages best.")
    st.caption("Tip: For speed, use smaller images and reduce max tokens.")

mode = st.radio("Mode", ["💬 Chat with one image", "📚 Batch summarise"], horizontal=True,
                label_visibility="collapsed")

BATCH_PROMPT = (
    "Look ONLY at the image. Describe it in 2-3 sentences: the main subject, the setting, "
    "and anything notable (text, people, actions). Plain prose, no lists."
)


def batch_csv(rows):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["file", "summary"])
    w.writerows(rows)
    return buf.getvalue()


def batch_markdown(rows):
    parts = [f"# Image summaries ({len(rows)} images)", f"_Model: {model} · Language: {language} · "
             f"{datetime.now():%Y-%m-%d %H:%M}_", ""]
    for name, summary in rows:
        parts += [f"## {name}", summary, ""]
    return "\n".join(parts)


if mode.startswith("📚"):
    files = st.file_uploader("Upload images (JPG/PNG) — up to 50", type=["jpg", "jpeg", "png"],
                             accept_multiple_files=True)
    if not files:
        st.info("Upload several images to get a short summary of each, then download them all as CSV or Markdown.")
        st.stop()
    files = files[:50]
    if st.button(f"📝 Summarise {len(files)} image{'s' if len(files) != 1 else ''}", type="primary"):
        rows = []
        progress = st.progress(0.0, text="Starting…")
        for i, f in enumerate(files, 1):
            progress.progress((i - 1) / len(files), text=f"{i}/{len(files)} · {f.name}")
            summary = safe_query(model, BATCH_PROMPT + language_instruction(language), f.getvalue(),
                                 min(300, max_tokens))
            if summary is None:  # Ollama down — error already shown, stop the batch
                break
            rows.append((f.name, summary))
        progress.progress(1.0, text=f"Done · {len(rows)} of {len(files)} summarised")
        st.session_state.batch_rows = rows

    rows = st.session_state.get("batch_rows") or []
    if rows:
        for (name, summary), f in zip(rows, files):
            with st.container(border=True):
                c1, c2 = st.columns([1, 3])
                c1.image(f.getvalue(), use_container_width=True)
                c2.markdown(f"**{name}**")
                c2.write(summary)
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ CSV", batch_csv(rows), file_name="image-summaries.csv", mime="text/csv",
                           use_container_width=True)
        d2.download_button("⬇️ Markdown", batch_markdown(rows), file_name="image-summaries.md",
                           mime="text/markdown", use_container_width=True)
    st.stop()

uploaded = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if not uploaded:
    st.info("Upload an image to begin.")
    st.stop()

image_bytes = uploaded.read()
image_id = hashlib.sha1(image_bytes).hexdigest()

# Reset the conversation whenever a new image is uploaded
if st.session_state.get("image_id") != image_id:
    st.session_state.image_id = image_id
    st.session_state.history = []
    st.session_state.summary = None

st.image(image_bytes, caption=uploaded.name, use_container_width=True)

c1, c2 = st.columns(2)
with c1:
    if st.button("📝 Summarize (2 paragraphs, ≥500 words)", use_container_width=True):
        with st.spinner("Summarizing..."):
            st.session_state.summary = safe_query(
                model, SUMMARY_PROMPT + language_instruction(language), image_bytes, max_tokens)
with c2:
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.summary = None

if st.session_state.summary:
    with st.expander("Summary", expanded=True):
        st.write(st.session_state.summary)
        st.caption(f"{len(st.session_state.summary.split())} words")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])

question = st.chat_input("Ask a question about the image…")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = safe_query(
                model,
                build_question_prompt(question, st.session_state.history) + language_instruction(language),
                image_bytes,
                min(400, max_tokens),
            )
        if answer:
            st.write(answer)
            st.session_state.history.append({"q": question, "a": answer})

if st.session_state.summary or st.session_state.history:
    st.download_button(
        "⬇️ Download session (Markdown)",
        transcript(uploaded.name, model, st.session_state.summary, st.session_state.history, language),
        file_name=f"{uploaded.name.rsplit('.', 1)[0]}-analysis.md",
        mime="text/markdown",
    )

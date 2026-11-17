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


def photo_info(raw: bytes) -> dict:
    """Camera, date, size and GPS from the image's EXIF metadata (empty values if absent)."""
    from PIL import ExifTags, Image

    info = {"size": "", "format": "", "camera": "", "taken": "", "software": "", "gps": None}
    try:
        img = Image.open(io.BytesIO(raw))
    except Exception:
        return info
    info["size"] = f"{img.width} × {img.height}"
    info["format"] = img.format or ""
    exif = img.getexif()
    if not exif:
        return info
    tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
    sub = exif.get_ifd(0x8769)  # Exif sub-IFD holds DateTimeOriginal
    taken = sub.get(36867) or tags.get("DateTime")
    info["camera"] = " ".join(str(x).strip() for x in (tags.get("Make"), tags.get("Model")) if x).strip()
    info["taken"] = str(taken or "").strip()
    info["software"] = str(tags.get("Software") or "").strip()
    gps = exif.get_ifd(0x8825)
    try:
        def deg(v, ref):
            d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
            return -d if ref in ("S", "W") else d
        if gps.get(2) and gps.get(4):
            info["gps"] = (round(deg(gps[2], gps.get(1)), 5), round(deg(gps[4], gps.get(3)), 5))
    except Exception:
        info["gps"] = None
    return info


def strip_metadata(raw: bytes) -> tuple[bytes, str]:
    """Re-encode the pixels only, dropping EXIF/GPS. Keeps orientation. Returns (bytes, extension)."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw)))
    out = io.BytesIO()
    if img.mode in ("RGBA", "LA", "P"):
        img.save(out, format="PNG")
        return out.getvalue(), "png"
    img.convert("RGB").save(out, format="JPEG", quality=95)
    return out.getvalue(), "jpg"


def query_ollama(model: str, prompt: str, image_bytes: bytes | None, num_predict: int = 900, timeout=300):
    """Ask the model about an image — or, with image_bytes=None, a text-only question."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": num_predict,
            "temperature": 0.4,
        },
    }
    if image_bytes is not None:
        payload["images"] = [base64.b64encode(image_bytes).decode("utf-8")]
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

mode = st.radio("Mode", ["💬 Chat with one image", "📚 Batch summarise", "🆚 Compare two images"],
                horizontal=True, label_visibility="collapsed")

# Compare works in two steps so it runs on every vision model (most small ones
# accept only one image per request): describe each image in a fixed structure,
# then compare the two descriptions with a text-only call.
COMPARE_DESCRIBE_PROMPT = (
    "Look ONLY at the image. List what you see in short lines under these headings: "
    "Objects (with colour and position), People and what they are doing, Text visible, "
    "Setting and lighting. Be specific and literal; do not guess."
)


def compare_prompt(desc_a: str, desc_b: str, focus: str = "") -> str:
    return (
        "Below are careful descriptions of two images, A and B.\n\n"
        f"IMAGE A:\n{desc_a}\n\nIMAGE B:\n{desc_b}\n\n"
        + (f"The user specifically wants to know: {focus.strip()}\n\n" if focus.strip() else "")
        + "Compare them. Reply in Markdown with exactly three sections:\n"
        "### Same in both\n- short bullet points\n"
        "### Different\n- short bullet points, each saying what A has vs what B has\n"
        "### In one sentence\nthe most important difference.\n"
        "Only use facts from the descriptions."
    )


def comparison_markdown(name_a, name_b, desc_a, desc_b, result):
    return "\n".join([
        f"# Comparison: {name_a} vs {name_b}",
        f"_Model: {model} · {datetime.now():%Y-%m-%d %H:%M}_", "",
        result, "", "---", f"## Image A — {name_a}", desc_a, "", f"## Image B — {name_b}", desc_b, "",
    ])


if mode.startswith("🆚"):
    ca, cb = st.columns(2)
    file_a = ca.file_uploader("Image A", type=["jpg", "jpeg", "png"], key="cmp_a")
    file_b = cb.file_uploader("Image B", type=["jpg", "jpeg", "png"], key="cmp_b")
    if file_a:
        ca.image(file_a.getvalue(), use_container_width=True)
    if file_b:
        cb.image(file_b.getvalue(), use_container_width=True)
    if not (file_a and file_b):
        st.info("Upload two images — before/after shots, two products, spot-the-difference puzzles…")
        st.stop()
    focus = st.text_input("Anything specific to compare? (optional)",
                          placeholder="e.g. what changed in the room, which outfit is more formal")
    pair_id = hashlib.sha1(file_a.getvalue() + b"|" + file_b.getvalue() + focus.encode()).hexdigest()
    if st.session_state.get("cmp_id") != pair_id:
        st.session_state.cmp_id = pair_id
        st.session_state.cmp = None
    if st.button("🆚 Compare", type="primary"):
        with st.status("Comparing…", expanded=True) as status:
            st.write("Looking at image A…")
            desc_a = safe_query(model, COMPARE_DESCRIBE_PROMPT, file_a.getvalue(), 350)
            desc_b = None
            if desc_a:
                st.write("Looking at image B…")
                desc_b = safe_query(model, COMPARE_DESCRIBE_PROMPT, file_b.getvalue(), 350)
            result = None
            if desc_a and desc_b:
                st.write("Comparing the two…")
                result = safe_query(model, compare_prompt(desc_a, desc_b, focus) + language_instruction(language),
                                    None, min(600, max_tokens))
            if result:
                st.session_state.cmp = (desc_a, desc_b, result)
                status.update(label="Done", state="complete", expanded=False)
            else:
                status.update(label="Comparison failed", state="error")
    if st.session_state.get("cmp"):
        desc_a, desc_b, result = st.session_state.cmp
        st.markdown(result)
        with st.expander("What the model saw in each image"):
            st.markdown(f"**Image A — {file_a.name}**\n\n{desc_a}\n\n**Image B — {file_b.name}**\n\n{desc_b}")
        st.download_button("⬇️ Download comparison (Markdown)",
                           comparison_markdown(file_a.name, file_b.name, desc_a, desc_b, result),
                           file_name="image-comparison.md", mime="text/markdown")
    st.stop()

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

meta = photo_info(image_bytes)
with st.expander("📷 Photo info & privacy" + ("  ·  ⚠️ contains GPS location" if meta["gps"] else "")):
    rows = {"Size": meta["size"], "Format": meta["format"], "Camera": meta["camera"],
            "Taken": meta["taken"], "Edited with": meta["software"]}
    st.markdown("\n".join(f"- **{k}:** {v}" for k, v in rows.items() if v) or "No metadata found.")
    if meta["gps"]:
        lat, lon = meta["gps"]
        st.warning(f"This photo reveals where it was taken ({lat}, {lon}). "
                   "Strip the metadata before sharing it publicly.")
        st.markdown(f"[View location on OpenStreetMap](https://www.openstreetmap.org/?mlat={lat}&mlon={lon}#map=16/{lat}/{lon})")
    if any(v for k, v in meta.items() if k not in ("size", "format")):
        clean, ext = strip_metadata(image_bytes)
        st.download_button("⬇️ Download a copy without metadata", clean,
                           file_name=f"{uploaded.name.rsplit('.', 1)[0]}-clean.{ext}",
                           mime="image/png" if ext == "png" else "image/jpeg")
    st.caption("Read on your computer — nothing is uploaded anywhere.")

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

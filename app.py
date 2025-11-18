import io
import json
import re
import streamlit as st
import requests
import pdfplumber
import fitz  # PyMuPDF
from groq import Groq

# ---- Quiet down PyMuPDF's PDF color warnings ----
try:
    fitz.TOOLS.set_verbosity(0)
except Exception:
    pass

# ---- Config / Secrets ----
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
GROQ_MODEL = st.secrets.get("GROQ_MODEL", "llama-3.1-8b-instant")  # current production model
N8N_WEBHOOK_URL = st.secrets.get("N8N_WEBHOOK_URL", "")

if not GROQ_API_KEY:
    st.error("Add GROQ_API_KEY to .streamlit/secrets.toml")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# ---- Utilities ----
def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Try pdfplumber first (nice layout), fall back to PyMuPDF."""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
            return "\n".join(pages)
    except Exception:
        text = []
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            for page in doc:
                # plain text is enough and avoids most warnings
                text.append(page.get_text("text"))
        return "\n".join(text)

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore")

def ask_groq_json(system: str, user_prompt: str) -> dict:
    """
    Call Groq Chat Completions in JSON mode and return parsed dict.
    """
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"}  # forces JSON output
    )
    content = resp.choices[0].message.content
    # Usually already JSON; keep a small fallback that finds the first {...} block.
    try:
        return json.loads(content)
    except Exception:
        m = re.search(r"\{.*\}", content, flags=re.S)
        if m:
            return json.loads(m.group(0))
        return {"raw": content}

# ---- Streamlit UI ----
st.set_page_config(page_title="AI-Powered Document Orchestrator — Groq", page_icon="📄", layout="centered")
st.title("📄 AI-Powered Document Orchestrator — Groq")

st.markdown("Upload a **PDF** or **TXT** and ask a question. The model returns structured JSON.")

uploaded = st.file_uploader("Upload document", type=["pdf", "txt"])
question = st.text_input("Your analytical question", "List the skills mentioned.")

if st.button("🔍 Extract", disabled=not (uploaded and question)):
    with st.spinner("Reading + extracting..."):
        raw = uploaded.read()
        text = extract_text_from_pdf(raw) if uploaded.name.lower().endswith(".pdf") else extract_text_from_txt(raw)

        system = "You are a precise JSON extraction engine. Return ONLY compact JSON. No extra text."
        user_prompt = f"""
Return JSON with the following shape:
{{
  "key_points": [{{"key": "…", "value": "…"}}],
  "risk_level": "Low|Medium|High",
  "confidence": 0.0
}}

Rules:
- Aim for 5–8 key_points if possible; fewer is OK if the document is short.
- Keep values concise.
- If 'risk' doesn't apply, set "Low".
- confidence ∈ [0,1].

QUESTION:
{question}

DOCUMENT TEXT (truncated if long):
{text[:20000]}
        """.strip()

        try:
            data = ask_groq_json(system, user_prompt)
            st.session_state["raw_text"] = text[:50000]
            st.session_state["question"] = question
            st.session_state["extracted_json"] = data
        except Exception as e:
            st.error(f"Extraction failed: {e}")

# Stage 2: show JSON
if "extracted_json" in st.session_state:
    st.subheader("Structured Data Extracted (JSON)")
    st.json(st.session_state["extracted_json"])

    # Optional: Email via n8n
    st.subheader("Send Conditional Alert Email (via n8n)")
    recip = st.text_input("Recipient Email ID", key="recipient_email")
    if st.button("📧 Send Alert Mail", disabled=not recip):
        if not N8N_WEBHOOK_URL:
            st.warning("N8N_WEBHOOK_URL is not set in secrets. Skipping call.")
        else:
            with st.spinner("Calling n8n webhook..."):
                payload = {
                    "question": st.session_state["question"],
                    "raw_text": st.session_state["raw_text"],
                    "extracted_json": st.session_state["extracted_json"],
                    "recipient_email": recip
                }
                try:
                    r = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=60)
                    try:
                        data = r.json()
                    except Exception:
                        data = {"status": f"HTTP {r.status_code}", "body": r.text}

                    st.subheader("Final Analytical Answer (from n8n)")
                    st.write(data.get("final_answer", "_No answer returned_"))

                    st.subheader("Generated Email Body")
                    st.write(data.get("email_body", "_No email body returned_"))

                    st.subheader("Email Automation Status")
                    st.success(data.get("status", "Unknown"))
                except Exception as e:
                    st.error(f"Webhook failed: {e}")

"""
Clinical Note Summarizer — Privacy-Safe RAG
============================================
100% Free Stack:
  spaCy NER      → PII anonymization (runs offline)
  sentence-transformers → local embeddings (no API)
  ChromaDB       → vector store (free, persistent)
  Groq free API  → LLM generation (sign up at groq.com)
  SQLite         → HIPAA audit log (no AWS needed)
  FastAPI        → backend
  Streamlit      → UI

Run: streamlit run app.py
"""

import streamlit as st
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from ingestion.pii_anonymizer import anonymize, get_pii_report
from ingestion.chunker import chunk_text
from embeddings.chroma_store import index_chunks, retrieve, get_collection_stats, clear_collection
from rag.generator import generate_answer
from audit.sqlite_logger import log_query, get_audit_log

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
  .title { font-size: 1.8rem; font-weight: 700; color: #1a3a5c; }
  .subtitle { color: #6c757d; margin-bottom: 1.5rem; font-size: 0.95rem; }
  .anon-box { background:#f0f7f0; border-left:4px solid #28a745;
              border-radius:6px; padding:0.8rem 1rem; margin:0.5rem 0; font-size:0.85rem; }
  .answer-box { background:#f8f9fa; border-left:4px solid #1a3a5c;
                border-radius:6px; padding:1rem 1.2rem; font-size:0.95rem; }
  .source-box { background:#fff3cd; border-radius:6px;
                padding:0.6rem 0.9rem; margin:0.3rem 0; font-size:0.82rem; }
  .pii-tag { display:inline-block; background:#ffe0e0; color:#a00;
             border-radius:4px; padding:1px 6px; font-size:0.78rem;
             font-weight:600; margin:1px; }
  .audit-row { font-size:0.8rem; border-bottom:1px solid #dee2e6; padding:4px 0; }
  .stat-num { font-size:1.4rem; font-weight:700; color:#1a3a5c; }
  .hipaa-badge { background:#d4edda; color:#155724; border-radius:20px;
                 padding:3px 12px; font-size:0.78rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown('<p class="title">🏥 Clinical Note RAG Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Privacy-safe Q&A over clinical notes · PII anonymized · HIPAA audit log · 100% free stack</p>', unsafe_allow_html=True)
with col_h2:
    st.markdown('<br><span class="hipaa-badge">🔒 HIPAA-Aware</span>', unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Free at groq.com — no credit card needed",
        placeholder="gsk_..."
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    groq_model = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768"],
        help="All free on Groq"
    )

    top_k = st.slider("Chunks to retrieve (top-k)", 2, 8, 4)

    st.markdown("---")
    st.markdown("**📊 Collection Stats**")
    stats = get_collection_stats()
    st.markdown(f'<p class="stat-num">{stats["count"]}</p><p style="font-size:0.8rem;color:#6c757d">chunks indexed</p>', unsafe_allow_html=True)

    if st.button("🗑️ Clear All Notes", type="secondary"):
        clear_collection()
        st.success("Collection cleared.")
        st.rerun()

    st.markdown("---")
    st.markdown("**Setup**")
    st.code("pip install -r requirements.txt\npython -m spacy download en_core_web_sm", language="bash")

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Upload Notes", "💬 Ask Questions", "📋 Audit Log"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Upload & Index
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload Clinical Notes")
    st.caption("Notes are anonymized before indexing — PII never enters the vector store.")

    input_method = st.radio("Input method", ["Paste text", "Upload .txt file"], horizontal=True)

    raw_text = ""
    doc_id = ""

    if input_method == "Paste text":
        doc_id = st.text_input("Document ID (e.g. patient-001)", value="note-001")
        raw_text = st.text_area(
            "Paste clinical note here",
            height=250,
            placeholder="Patient John Smith, DOB 01/15/1985, MRN 123456...\nChief complaint: chest pain for 3 days..."
        )
    else:
        doc_id = st.text_input("Document ID", value="note-001")
        uploaded = st.file_uploader("Upload .txt clinical note", type=["txt"])
        if uploaded:
            raw_text = uploaded.read().decode("utf-8")
            st.text_area("File preview", raw_text[:500] + ("..." if len(raw_text) > 500 else ""), height=150, disabled=True)

    if st.button("🔒 Anonymize & Index", type="primary", disabled=not raw_text):
        with st.spinner("Detecting PII with spaCy..."):
            anon_text, pii_found = anonymize(raw_text)
            pii_report = get_pii_report(raw_text)

        # Show PII detected
        st.markdown("**PII Detected & Removed:**")
        if pii_report:
            pii_html = " ".join(f'<span class="pii-tag">{label}: {text}</span>'
                                for text, label in pii_report)
            st.markdown(f'<div class="anon-box">✅ Anonymized {len(pii_report)} PII entities<br>{pii_html}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="anon-box">✅ No PII detected</div>', unsafe_allow_html=True)

        col_raw, col_anon = st.columns(2)
        with col_raw:
            st.caption("Original (contains PII)")
            st.text_area("", raw_text[:400], height=150, disabled=True, label_visibility="collapsed")
        with col_anon:
            st.caption("Anonymized (safe to index)")
            st.text_area("", anon_text[:400], height=150, disabled=True, label_visibility="collapsed")

        with st.spinner("Chunking & embedding..."):
            chunks = chunk_text(anon_text, doc_id=doc_id)
            index_chunks(chunks)

        st.success(f"✅ Indexed {len(chunks)} chunks from '{doc_id}' into ChromaDB.")

    # Sample note for demo
    with st.expander("📋 Load sample clinical note for demo"):
        sample = """Patient: John Smith
Date of Birth: March 15, 1978
MRN: 78234-A
SSN: 123-45-6789
Address: 456 Oak Street, Cincinnati, OH 45202
Phone: (513) 555-0192
Attending Physician: Dr. Sarah Johnson

Chief Complaint: Patient presents with persistent chest pain radiating to the left arm for the past 72 hours.

History of Present Illness:
Mr. Smith is a 46-year-old male with a history of hypertension and type 2 diabetes mellitus. He reports substernal chest pain rated 7/10 in severity, onset 3 days ago while climbing stairs. Pain is described as pressure-like, radiating to the left arm and jaw. Associated symptoms include diaphoresis and mild shortness of breath. Patient denies nausea or vomiting. He took two aspirin tablets at home with minimal relief.

Past Medical History:
- Hypertension (diagnosed 2015, well-controlled on lisinopril 10mg daily)
- Type 2 Diabetes Mellitus (HbA1c 7.2% at last visit, on metformin 1000mg twice daily)
- Hyperlipidemia (on atorvastatin 40mg nightly)

Physical Examination:
Vital Signs: BP 148/92 mmHg, HR 88 bpm, RR 18/min, SpO2 97% on room air, Temp 98.6°F
Cardiovascular: Regular rate and rhythm, no murmurs, rubs, or gallops
Respiratory: Clear to auscultation bilaterally
Abdomen: Soft, non-tender, non-distended

Laboratory Results:
- Troponin I: 0.8 ng/mL (elevated, ref <0.04)
- BNP: 145 pg/mL (mildly elevated)
- CBC: WBC 9.2, Hgb 13.8, Plt 224
- CMP: Na 138, K 4.1, Cr 0.9, Glucose 142

ECG: ST elevation in leads V2-V5, reciprocal changes in inferior leads

Assessment: STEMI — ST-elevation myocardial infarction

Plan:
1. Emergent cardiac catheterization
2. Dual antiplatelet therapy: aspirin 325mg + ticagrelor 180mg loading dose
3. Heparin infusion per weight-based protocol
4. Cardiology consult — Dr. Michael Chen paged
5. ICU admission for continuous monitoring"""

        if st.button("Load Sample Note"):
            st.session_state["sample_loaded"] = sample
            st.rerun()

    if "sample_loaded" in st.session_state:
        st.info("Sample note loaded! Copy the text above into the 'Paste text' input and click Anonymize & Index.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Ask Questions
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Ask Questions About Clinical Notes")

    if get_collection_stats()["count"] == 0:
        st.warning("No notes indexed yet. Go to the Upload Notes tab to add some first.")
    else:
        question = st.text_input(
            "Clinical question",
            placeholder="What medications is the patient currently taking?",
        )

        example_qs = [
            "What are the patient's vital signs?",
            "What medications is the patient on?",
            "What is the diagnosis and treatment plan?",
            "What lab results were abnormal?",
            "What is the patient's medical history?",
        ]
        st.caption("Example questions:")
        cols = st.columns(len(example_qs))
        for i, eq in enumerate(example_qs):
            if cols[i].button(eq, key=f"eq_{i}", use_container_width=True):
                question = eq

        if st.button("🔍 Get Answer", type="primary", disabled=not question):
            if not os.environ.get("GROQ_API_KEY"):
                st.error("Please enter your Groq API key in the sidebar. Free at groq.com")
            else:
                # Retrieve
                with st.spinner("Retrieving relevant chunks..."):
                    chunks = retrieve(question, top_k=top_k)

                if not chunks:
                    st.warning("No relevant chunks found. Try indexing some notes first.")
                else:
                    # Generate
                    with st.spinner(f"Generating answer with {groq_model}..."):
                        result = generate_answer(
                            question=question,
                            chunks=chunks,
                            model=groq_model
                        )

                    # Answer
                    st.markdown("**📋 Answer:**")
                    st.markdown(f'<div class="answer-box">{result["answer"]}</div>',
                                unsafe_allow_html=True)

                    # Sources
                    st.markdown("**📎 Source Passages Used:**")
                    for i, chunk in enumerate(chunks):
                        st.markdown(
                            f'<div class="source-box"><strong>Source {i+1}</strong> '
                            f'(doc: {chunk["doc_id"]}, chunk: {chunk["chunk_id"]})<br>'
                            f'{chunk["text"][:200]}{"..." if len(chunk["text"]) > 200 else ""}</div>',
                            unsafe_allow_html=True
                        )

                    # Confidence
                    conf = result.get("confidence", "medium")
                    conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf, "🟡")
                    st.caption(f"{conf_color} Confidence: {conf} · {len(chunks)} chunks retrieved")

                    # Log to audit
                    log_query(
                        question=question,
                        answer=result["answer"],
                        sources=[f"{c['doc_id']}:chunk{c['chunk_id']}" for c in chunks],
                        model=groq_model,
                        confidence=conf
                    )
                    st.caption("✅ Query logged to HIPAA audit trail")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Audit Log
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("HIPAA Audit Log")
    st.caption("Every query and response is logged with timestamp for compliance.")

    logs = get_audit_log(limit=50)

    if not logs:
        st.info("No queries logged yet.")
    else:
        st.success(f"{len(logs)} queries logged")

        # Download
        import json
        st.download_button(
            "📥 Download Full Audit Log (JSON)",
            data=json.dumps(logs, indent=2),
            file_name="hipaa_audit_log.json",
            mime="application/json"
        )

        for entry in logs:
            with st.expander(f"[{entry['timestamp']}] {entry['question'][:80]}"):
                st.markdown(f"**Question:** {entry['question']}")
                st.markdown(f"**Answer:** {entry['answer']}")
                st.markdown(f"**Sources:** {entry['sources']}")
                st.markdown(f"**Model:** {entry['model']} · **Confidence:** {entry['confidence']}")

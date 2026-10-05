import os
import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any

# Page Configuration
st.set_page_config(
    page_title="Enterprise Agentic AI Research Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Branding */
    .brand-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #111119 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px 28px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .brand-title {
        font-size: 26px;
        font-weight: 700;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .brand-sub {
        color: #94a3b8;
        font-size: 14px;
        margin: 0;
    }

    /* Badges */
    .route-badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-right: 8px;
        margin-bottom: 12px;
    }
    .badge-rag {
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .badge-research {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-both {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    /* Card styling */
    .source-card {
        background: #181824;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .source-title {
        font-size: 13px;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 4px;
    }
    .source-text {
        font-size: 12px;
        color: #94a3b8;
        line-height: 1.5;
    }

    /* Circuit status pills */
    .pill-closed {
        color: #10b981;
        font-weight: 600;
        background: rgba(16, 185, 129, 0.1);
        padding: 3px 10px;
        border-radius: 6px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .pill-open {
        color: #ef4444;
        font-weight: 600;
        background: rgba(239, 68, 68, 0.1);
        padding: 3px 10px;
        border-radius: 6px;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Central API URL
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

# Helper Functions
def fetch_health() -> Dict[str, Any]:
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"status": "offline", "circuit_breakers": {}}

def fetch_documents() -> List[Dict[str, Any]]:
    try:
        r = requests.get(f"{BACKEND_URL}/documents", timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def delete_document_api(doc_id: str) -> bool:
    try:
        r = requests.delete(f"{BACKEND_URL}/documents/{doc_id}", timeout=10)
        return r.status_code == 200
    except Exception:
        return False

def upload_files_api(files) -> List[Dict[str, Any]]:
    files_payload = [("files", (f.name, f.getvalue(), f.type)) for f in files]
    try:
        r = requests.post(f"{BACKEND_URL}/documents/upload", files=files_payload, timeout=60)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"Upload connection error: {e}")
    return []

def ask_question_api(query: str, doc_ids: List[str] = None) -> Dict[str, Any]:
    payload = {"query": query, "selected_document_ids": doc_ids or None}
    try:
        r = requests.post(f"{BACKEND_URL}/ask", json=payload, timeout=120)
        if r.status_code == 200:
            return r.json()
        else:
            return {"final_answer": f"Backend API returned error {r.status_code}: {r.text}", "route_taken": "error"}
    except Exception as e:
        return {"final_answer": f"Could not connect to FastAPI backend at {BACKEND_URL}. Ensure it is running. Error: {e}", "route_taken": "error"}

# Sidebar: Reliability Status & Document Management
with st.sidebar:
    st.markdown("### ⚙️ System Status")
    health = fetch_health()
    server_online = health.get("status") == "healthy"
    
    if server_online:
        st.success("🟢 FastAPI Backend: Online")
    else:
        st.error("🔴 FastAPI Backend: Offline (Port 8000)")

    # Live Circuit Breakers Display with intuitive labels
    cb_statuses = health.get("circuit_breakers", {})
    st.markdown("#### 🛡️ Service Circuit Breakers")
    st.caption("🟢 Active = Healthy | 🔴 Tripped = In Fallback")
    cols = st.columns(2)
    services = ["milvus", "gemini", "reranker", "tavily"]
    for i, s in enumerate(services):
        col = cols[i % 2]
        s_data = cb_statuses.get(s, {})
        
        # User-friendly terminology
        if not server_online:
            label = "⚪ Offline"
            pill_class = "pill-open"
        else:
            state = s_data.get("state", "CLOSED")
            if state == "CLOSED":
                label = "🟢 Active"
                pill_class = "pill-closed"
            elif state == "OPEN":
                label = "🔴 Tripped"
                pill_class = "pill-open"
            else:
                label = "🟡 Recovering"
                pill_class = "pill-open"

        with col:
            st.markdown(f"**{s.capitalize()}**<br><span class='{pill_class}'>{label}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📁 Document Scope & Upload")

    # Document Upload Dropzone
    uploaded_files = st.file_uploader(
        "Upload Internal Knowledge",
        type=["pdf", "docx", "txt", "csv", "xlsx", "pptx", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("🚀 Process & Ingest Documents", use_container_width=True):
            with st.spinner("Parsing, chunking, and embedding documents..."):
                results = upload_files_api(uploaded_files)
                for res in results:
                    if res.get("status") == "success":
                        st.toast(f"✅ {res.get('document_name')}: {res.get('chunk_count')} chunks indexed!")
                    else:
                        st.error(f"❌ {res.get('document_name')}: {res.get('message')}")
                st.rerun()

    # List Indexed Documents
    docs = fetch_documents()
    st.markdown(f"**Indexed Documents ({len(docs)})**")
    
    doc_options = {}
    for d in docs:
        label = f"{d['document_name']} ({d['chunk_count']} chunks)"
        doc_options[label] = d["document_id"]

    selected_doc_labels = st.multiselect(
        "Filter Search Scope (Optional):",
        options=list(doc_options.keys()),
        help="Leave empty to search across ALL indexed documents, or select specific files."
    )
    selected_doc_ids = [doc_options[label] for label in selected_doc_labels]

    if docs:
        with st.expander("Manage Stored Files"):
            for d in docs:
                c1, c2 = st.columns([3, 1])
                c1.write(f"📄 **{d['document_name']}** ({d['file_type']})")
                if c2.button("🗑️", key=f"del_{d['document_id']}", help=f"Delete {d['document_name']}"):
                    if delete_document_api(d["document_id"]):
                        st.success(f"Removed {d['document_name']}")
                        st.rerun()

# Main Application Header
st.markdown("""
<div class="brand-header">
    <div class="brand-title">Agentic AI Research & Knowledge Assistant</div>
    <div class="brand-sub">
        LangGraph Multi-Agent Orchestration • Hybrid RAG (Milvus + BM25 + RRF + Cross-Encoder) • Tavily Live Web Research • 4-Tier Circuit Breakers
    </div>
</div>
""", unsafe_allow_html=True)

# Session state initialization for question input and current response
if "query_text_input" not in st.session_state:
    st.session_state["query_text_input"] = ""
if "current_response" not in st.session_state:
    st.session_state["current_response"] = None
if "current_elapsed" not in st.session_state:
    st.session_state["current_elapsed"] = 0.0

def set_query_callback(query_text: str):
    st.session_state["query_text_input"] = query_text

def clear_query_callback():
    st.session_state["query_text_input"] = ""
    st.session_state["current_response"] = None
    st.session_state["current_elapsed"] = 0.0

# Example Queries
st.markdown("**💡 Quick Examples:**")
eq_cols = st.columns(3)
with eq_cols[0]:
    st.button(
        "🏢 Internal: Types of leaves in handbook?",
        on_click=set_query_callback,
        args=("What are the different types of leaves available in our company?",),
        use_container_width=True
    )
with eq_cols[1]:
    st.button(
        "🌐 External: Who is the CEO of Google?",
        on_click=set_query_callback,
        args=("Who is the current CEO of Google and Alphabet?",),
        use_container_width=True
    )
with eq_cols[2]:
    st.button(
        "🔢 Math: Calculate 12 + 12 = ?",
        on_click=set_query_callback,
        args=("12+12=?",),
        use_container_width=True
    )

# Query Input Box bound directly to session state key
user_query = st.text_area(
    "Enter research inquiry or question:",
    key="query_text_input",
    placeholder="e.g. What are the different types of leaves available? Or ask general/math queries like 12+12=?",
    height=85
)

# Action buttons: Execute Query & Clear
btn_col1, btn_col2, btn_col3 = st.columns([2, 2, 6])
with btn_col1:
    submit_button = st.button("🔎 Execute Query", type="primary", use_container_width=True)
with btn_col2:
    st.button("🧹 Clear", on_click=clear_query_callback, use_container_width=True)

# Submit button handler
if submit_button and user_query.strip():
    with st.spinner("Supervisor analyzing query & executing agent workflow..."):
        start_time = time.time()
        response = ask_question_api(user_query.strip(), selected_doc_ids)
        elapsed = time.time() - start_time
        st.session_state["current_response"] = response
        st.session_state["current_elapsed"] = elapsed

# Display the response if present in session state
response = st.session_state.get("current_response")
if response:
    elapsed = st.session_state.get("current_elapsed", 0.0)
    route = response.get("route_taken", "rag")
    badge_class = "badge-rag"
    badge_label = "🏢 Internal RAG Agent"
    if route == "research":
        badge_class = "badge-research"
        badge_label = "🌐 External Research Agent"
    elif route == "both":
        badge_class = "badge-both"
        badge_label = "🤝 Collaborative: Internal RAG + External Research"
    elif route == "blocked_by_guardrails":
        badge_class = "badge-rag"
        badge_label = "🛡️ Blocked by Input Guardrail"

    # Display Route and Diagnostics
    is_cached_resp = response.get('cached', False)
    cache_badge_html = '<span style="background: rgba(16, 185, 129, 0.25); color: #34d399; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; margin-left: 8px;">⚡ Cache HIT</span>' if is_cached_resp else '<span style="background: rgba(99, 102, 241, 0.2); color: #818cf8; padding: 3px 8px; border-radius: 6px; font-weight: 600; font-size: 12px; margin-left: 8px;">🔄 Processed by Worker</span>'
    st.markdown(f"""
    <span class="route-badge {badge_class}">{badge_label}</span>
    {cache_badge_html}
    <span style="color:#94a3b8; font-size:13px; margin-left: 10px;">⏱️ Latency: {response.get('execution_time_seconds', round(elapsed, 2))}s | 🛡️ Guardrails: {'Passed' if response.get('guardrails_passed', True) else 'Flagged'}</span>
    """, unsafe_allow_html=True)

    if response.get("supervisor_reasoning"):
        st.info(f"**Supervisor Rationale:** {response.get('supervisor_reasoning')}")

    # Display Synthesized Final Answer
    st.markdown("### 📋 Executive Response")
    st.markdown(response.get("final_answer", "No response returned."))

    # Citations & Evidence Drawers
    int_sources = response.get("internal_sources", [])
    ext_sources = response.get("external_sources", [])

    if int_sources or ext_sources:
        st.markdown("---")
        st.markdown("### 📚 Grounded Sources & Citations")
        
        tab1, tab2 = st.tabs([f"Internal Documents ({len(int_sources)})", f"External Web Research ({len(ext_sources)})"])
        
        with tab1:
            if int_sources:
                for idx, src in enumerate(int_sources, 1):
                    score_str = f" • Rerank Score: {src.get('score', 'N/A')}" if src.get("score") is not None else ""
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">📄 [{idx}] {src.get('document_name')} ({src.get('page_or_section')}){score_str}</div>
                        <div class="source-text">{src.get('snippet')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No internal documents were retrieved for this response.")

        with tab2:
            if ext_sources:
                for idx, src in enumerate(ext_sources, 1):
                    url_link = f"<a href='{src.get('url')}' target='_blank' style='color:#60a5fa;'>{src.get('url')}</a>" if src.get("url") else "No URL"
                    st.markdown(f"""
                    <div class="source-card">
                        <div class="source-title">🌐 [{idx}] {src.get('title')} ({url_link})</div>
                        <div class="source-text">{src.get('snippet')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption("No external web research was required for this response.")

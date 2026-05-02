# TODO: Streamlit UI
"""
app.py — Streamlit UI for CrimeScope-AI
-----------------------------------------
Features:
  - Crime description input
  - Live "thinking" display (ChatGPT-style step-by-step)
  - Structured results as cards
  - Latency display

Run:
  streamlit run app.py
"""

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CrimeScope AI",
    page_icon="⚖️",
    layout="wide",
)

import os
API_URL = os.getenv("API_URL", "http://localhost:8000/analyze-crime")
# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("⚖️ CrimeScope AI")
st.caption("RAG-Powered Indian Legal Advisor — Identify applicable laws from crime descriptions")
st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    description = st.text_area(
        label="📝 Describe the Crime Scene",
        placeholder=(
            "Example: A person broke into a house at night, "
            "threatened the owner with a knife, and stole valuables worth ₹50,000..."
        ),
        height=150,
    )

with col2:
    st.markdown("### 📌 Tips")
    st.info(
        "- Describe what happened in detail\n"
        "- Mention weapons, intent, victim, location\n"
        "- Multiple crimes in one description are supported\n"
        "- English or Hindi both work"
    )

analyze_btn = st.button("🔍 Analyze Crime", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if analyze_btn:
    if not description or len(description.strip()) < 20:
        st.warning("Please describe the crime in at least 20 characters.")
        st.stop()

    # Live thinking display — ChatGPT style
    with st.status("⚖️ Analyzing crime scenario...", expanded=True) as status:
        st.write("🔍 Understanding the crime description...")
        st.write("📝 Generating search queries with Multi-Query Retrieval (Groq)...")
        st.write("📚 Searching legal database (ChromaDB)...")
        st.write("🤖 Generating legal analysis with Groq(llama-3.3-70b-versatile)...")

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    API_URL,
                    json={"description": description.strip()},
                )
            response.raise_for_status()
            data = response.json()
            status.update(label=" Analysis complete!", state="complete")

        except httpx.ConnectError:
            status.update(label=" Cannot connect to API", state="error")
            st.error("FastAPI server is not running. Start it with: `uvicorn api:app --reload`")
            st.stop()
        except Exception as e:
            status.update(label=" Analysis failed", state="error")
            st.error(f"Error: {str(e)}")
            st.stop()

    # ---------------------------------------------------------------------------
    # Results
    # ---------------------------------------------------------------------------
    if data.get("success"):
        analysis = data["analysis"]
        elapsed  = data.get("processing_time_seconds", 0)

        # Metrics row
        m1, m2, m3 = st.columns(3)
        m1.metric("⚡ Response Time", f"{elapsed:.2f}s")
        m2.metric("🔍 Crime Types Found", len(analysis["crime_type"]))
        m3.metric("📖 Laws Identified", len(analysis["applicable_laws"]))

        st.divider()

        # Crime types as badges
        st.markdown("### 🚨 Identified Crime Types")
        badge_html = " ".join(
            f'<span style="background:#dc2626;color:white;padding:4px 12px;'
            f'border-radius:20px;margin:4px;display:inline-block;font-weight:600">{ct}</span>'
            for ct in analysis["crime_type"]
        )
        st.markdown(badge_html, unsafe_allow_html=True)

        # Ambiguity note
        if analysis.get("ambiguity_note"):
            st.warning(f" **Note:** {analysis['ambiguity_note']}")

        st.divider()

        # Applicable laws as cards
        st.markdown("### 📋 Applicable Legal Provisions")

        for i, law in enumerate(analysis["applicable_laws"], 1):
            with st.expander(
                f"**{i}. {law['act']} — Section {law['section']}**",
                expanded=(i <= 3),   # First 3 open by default
            ):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**📖 What this section says:**")
                    st.info(law["description"])
                with col_b:
                    st.markdown("**⚖️ Why it applies here:**")
                    st.success(law["justification"])

        # Raw JSON (for developers / assignment submission)
        with st.expander("🔧 Raw JSON Output"):
            st.json(analysis)

    else:
        st.error(f"Analysis failed: {data.get('error', 'Unknown error')}")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.caption("CrimeScope AI | Built with LangChain + Groq(llama-3.3-70b-versatile) + ChromaDB | Digixito Assignment")

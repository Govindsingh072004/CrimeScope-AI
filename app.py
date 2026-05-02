"""
app.py — CrimeScope-AI Streamlit UI (No FastAPI needed)
"""
import streamlit as st
from src.chain import analyze_crime

st.set_page_config(page_title="CrimeScope AI", page_icon="⚖️", layout="wide")
st.title("⚖️ CrimeScope AI")
st.caption("RAG-Powered Indian Legal Advisor — Identify applicable laws from crime descriptions")
st.divider()

col1, col2 = st.columns([2, 1])
with col1:
    description = st.text_area(
        label="📝 Describe the Crime Scene",
        placeholder="Example: A person broke into a house at night, threatened the owner with a knife...",
        height=150,
    )
with col2:
    st.markdown("### 📌 Tips")
    st.info("- Describe what happened in detail\n- Mention weapons, intent, victim, location\n- Multiple crimes supported\n- English or Hindi both work")

analyze_btn = st.button("🔍 Analyze Crime", type="primary", use_container_width=True)

if analyze_btn:
    if not description or len(description.strip()) < 20:
        st.warning("Please describe the crime in at least 20 characters.")
        st.stop()

    with st.status("⚖️ Analyzing crime scenario...", expanded=True) as status:
        st.write("🔍 Understanding the crime description...")
        st.write("📝 Generating search queries with Multi-Query Retrieval (Groq)...")
        st.write("📚 Searching legal database (ChromaDB)...")
        st.write("🤖 Generating legal analysis with Groq(llama-3.3-70b-versatile)...")
        try:
            result, elapsed, _ = analyze_crime(description=description.strip())
            status.update(label="✅ Analysis complete!", state="complete")
        except Exception as e:
            status.update(label="❌ Analysis failed", state="error")
            st.error(f"Error: {str(e)}")
            st.stop()

    m1, m2, m3 = st.columns(3)
    m1.metric("⚡ Response Time", f"{elapsed:.2f}s")
    m2.metric("🔍 Crime Types Found", len(result.crime_type))
    m3.metric("📖 Laws Identified", len(result.applicable_laws))

    st.divider()
    st.markdown("### 🚨 Identified Crime Types")
    badge_html = " ".join(
        f'<span style="background:#e74c3c;color:white;padding:4px 12px;border-radius:20px;margin:4px">{ct}</span>'
        for ct in result.crime_type
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    if result.ambiguity_note:
        st.warning(f"⚠️ **Note:** {result.ambiguity_note}")

    st.divider()
    st.markdown("### 📋 Applicable Legal Provisions")
    for i, law in enumerate(result.applicable_laws, 1):
        with st.expander(f"**{i}. {law.act} — Section {law.section}**", expanded=(i <= 3)):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**📖 What this section says:**")
                st.info(law.description)
            with col_b:
                st.markdown("**⚖️ Why it applies here:**")
                st.success(law.justification)

    with st.expander("🔧 Raw JSON Output"):
        st.json(result.model_dump())

st.divider()
st.caption("CrimeScope AI | Built with LangChain + Groq + ChromaDB | Digixito Assignment")

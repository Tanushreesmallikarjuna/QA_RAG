import streamlit as st
import requests

BACKEND_URL = "https://studymate-ai-backend-a9ms.onrender.com"

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="wide")

# ---------------- Theme toggle ----------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if st.session_state.dark_mode:
    bg_color = "#0F172A"
    card_color = "#1E293B"
    text_color = "#F1F5F9"
    accent = "#6366F1"
else:
    bg_color = "#F8FAFC"
    card_color = "#FFFFFF"
    text_color = "#1E293B"
    accent = "#4F46E5"

st.markdown(f"""
<style>
.stApp {{ background-color: {bg_color}; color: {text_color}; width:90%;}}
.block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
.stButton>button {{
    width: 100%;
    background: {accent};
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    
}}
.stButton>button:hover {{ background: {accent}; opacity: 0.9; }}
.stTextInput input {{ border: 1px solid {accent}; border-radius: 8px; }}
[data-testid="stFileUploader"] {{
    border: 1px dashed {accent};
    border-radius: 8px;
}}
.stSuccess, .stInfo, .stWarning {{ border-radius: 8px; }}
[data-testid="stSidebar"] {{ background-color: {card_color}; }}
</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar: conversation history ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    st.subheader("Conversation History")

    theme_label = "Switch to Light Mode" if st.session_state.dark_mode else "Switch to Dark Mode"
    if st.button(theme_label):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.divider()

    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {chat['question']}")
            st.markdown(f"**Bot:** {chat['answer']}")
            st.divider()

        if st.button("Clear Conversation"):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.caption("No conversation yet")

# ---------------- Main area ----------------
st.title("StudyMate AI")
st.caption("Ask questions across multiple PDFs using AI")

st.subheader("Step 1: Upload PDF(s)")
uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

if uploaded_file and st.button("Upload & Index PDF"):
    with st.spinner("Reading PDF and building index..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        r = requests.post(f"{BACKEND_URL}/upload", files=files)
    if r.status_code == 200:
        d = r.json()
        st.success(f"{d['message']} ({d['chunks_created']} chunks)")
        st.rerun()
    else:
        st.error(r.json()["detail"])

st.caption("Upload multiple PDFs one at a time. They will all be searchable together.")

st.subheader("Step 2: Ask a Question")
question = st.text_input("Ask a question about your PDFs",height=100)

if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching your documents..."):
            r = requests.post(
                f"{BACKEND_URL}/ask",
                json={"question": question, "history": st.session_state.chat_history}
            )
        if r.status_code == 200:
            data = r.json()
            st.write(data["answer"])
            st.session_state.chat_history.append({"question": question, "answer": data["answer"]})

            with st.expander("View Sources Used"):
                seen = set()
                shown = 0
                for src in data["sources"]:
                    if src["file"] in seen:
                        continue
                    seen.add(src["file"])
                    shown += 1
                    page = src.get("page", "?")
                    st.markdown(f"**Source {shown}:** `{src['file']}` (Page {page})")
                    preview = src["text"][:500] + "..." if len(src["text"]) > 500 else src["text"]
                    st.text(preview)
        else:
            st.error(r.json()["detail"])

# ---------------- Manage files (moved to bottom) ----------------
st.divider()
st.subheader("Manage Uploaded PDFs")

try:
    status = requests.get(f"{BACKEND_URL}/status").json()
    if status["total_pdfs"] > 0:
        st.caption(f"{status['total_pdfs']} PDF(s) indexed — {status['total_chunks']} total chunks")
        for file in status["files"]:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.write(file)
            with c2:
                if st.button("Delete", key=file):
                    r = requests.delete(f"{BACKEND_URL}/delete/{file}")
                    if r.status_code == 200:
                        st.success(r.json()["message"])
                        st.rerun()
                    else:
                        st.error(r.json()["detail"])
        st.divider()
        if st.button("Clear All PDFs"):
            r = requests.delete(f"{BACKEND_URL}/clear")
            if r.status_code == 200:
                st.success(r.json()["message"])
                st.rerun()
            else:
                st.error(r.json()["detail"])
    else:
        st.caption("No PDFs uploaded yet")
except:
    st.warning("Backend not reachable")

st.divider()
st.caption("Built with Streamlit, FastAPI, FAISS, fastembed, and Groq")
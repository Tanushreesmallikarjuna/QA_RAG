# Complete app.py
import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="StudyMate AI", page_icon="📚", layout="centered")

st.title("📚 StudyMate AI")
st.caption("Ask questions across multiple PDFs using AI")

st.markdown("""
<style>
.stApp{background-color:#F8FAFC;}
.block-container{padding-top:2rem;padding-bottom:2rem;}
.stButton>button{
 width:100%;
 background:#4F46E5;
 color:white;
 border:none;
 border-radius:10px;
 font-weight:600;
}
.stButton>button:hover{background:#4338CA;color:white;}
.stTextInput input{border:2px solid #4F46E5;border-radius:10px;}
[data-testid="stFileUploader"]{
 border:2px dashed #4F46E5;
 border-radius:10px;
 box-shadow:0 1px 3px rgba(0,0,0,0.08);
}
.stSuccess,.stInfo,.stWarning{border-radius:10px;}
</style>
""", unsafe_allow_html=True)

try:
    status = requests.get(f"{BACKEND_URL}/status").json()
    if status["total_pdfs"]>0:
        st.caption(f"📊 {status['total_pdfs']} PDF(s) indexed — {status['total_chunks']} total chunks")
        with st.expander("📂 Manage Uploaded PDFs"):
            for file in status["files"]:
                c1,c2=st.columns([5,1])
                with c1:
                    st.write(f"📄 {file}")
                with c2:
                    if st.button("🗑️ Delete",key=file):
                        r=requests.delete(f"{BACKEND_URL}/delete/{file}")
                        if r.status_code==200:
                            st.success(r.json()["message"]); st.rerun()
                        else:
                            st.error(r.json()["detail"])
            st.divider()
            if st.button("🗑️ Clear All PDFs"):
                r=requests.delete(f"{BACKEND_URL}/clear")
                if r.status_code==200:
                    st.success(r.json()["message"]); st.rerun()
                else:
                    st.error(r.json()["detail"])
    else:
        st.caption("No PDFs uploaded yet")
except:
    st.warning("⚠️ Backend not reachable")

st.subheader("📤 Step 1: Upload PDF(s)")
uploaded_file=st.file_uploader("Choose a PDF",type="pdf")
if uploaded_file and st.button("📤 Upload & Index PDF"):
    with st.spinner("📄 Reading PDF and building AI index..."):
        files={"file":(uploaded_file.name,uploaded_file.getvalue(),"application/pdf")}
        r=requests.post(f"{BACKEND_URL}/upload",files=files)
    if r.status_code==200:
        d=r.json()
        st.success(f"✅ {d['message']} ({d['chunks_created']} chunks)")
        st.rerun()
    else:
        st.error(r.json()["detail"])

st.info("💡 Upload multiple PDFs one at a time. They will all be searchable together.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

st.subheader("❓ Step 2: Ask Anything")
question=st.text_input("Ask a question about your PDFs")

if st.button("🚀 Ask Question"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🧠 Searching your documents..."):
            r=requests.post(f"{BACKEND_URL}/ask",json={"question":question,"history":st.session_state.chat_history})
        if r.status_code==200:
            data=r.json()
            st.write(data["answer"])
            st.session_state.chat_history.append({"question":question,"answer":data["answer"]})
            with st.expander("📚 View Sources Used"):
                seen=set(); shown=0
                for src in data["sources"]:
                    if src["file"] in seen:
                        continue
                    seen.add(src["file"]); shown+=1
                    page=src.get("page","?")
                    st.markdown(f"**Source {shown}:** 📄 `{src['file']}` (Page {page})")
                    preview=src["text"][:500]+"..." if len(src["text"])>500 else src["text"]
                    st.text(preview)
        else:
            st.error(r.json()["detail"])

if st.session_state.chat_history:
    st.subheader("💬 Conversation")
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {chat['question']}")
        st.markdown(f"**Bot:** {chat['answer']}")
        st.divider()
    if st.button("🗑️ Clear Conversation"):
        st.session_state.chat_history=[]
        st.rerun()

st.divider()
st.caption("Built with ❤️ using Streamlit • FastAPI • FAISS • Sentence Transformers • Groq")

#Finzalized for deployment
#with prompts, temp = 0 for deterministic output, message dipslays from privoded context only
# ----------------- IMPORTS -----------------

import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv


# --- LANGCHAIN IMPORTS ---
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


# --- DATABASE IMPORTS ---
from database import save_message, load_messages, clear_chat, get_chat_sessions


# ----------------- SETTINGS -----------------
load_dotenv()
st.set_page_config(page_title="📄 RAG Chatbot", layout="wide")
#  STYLING
# --- LOAD CUSTOM CSS ---
def load_css():
    st.markdown("""
        <style>
            .stApp { background-color: #F0F2F6; }


            /* Sidebar */
            [data-testid="stSidebar"] {
                background-color: #1E1E2E;
                color: #FFFFFF;
                padding: 10px;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
            }


            /* DOC-BOT Title */
            .sidebar-title {
                font-size: 3rem;
                font-weight: 1000;
                color: #FFFFFF;
                margin: 5px 0 20px 0;
                text-align: left-center;
            }


            /* Sidebar Section Wrappers */
            .sidebar-section {
                background-color: #2A2A3C;
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 15px;
                max-height: 220px;
                overflow-y: auto;
            }


            /* Upload text fix */
            .sidebar-section label {
                color: #5302cc!important;
                font-weight: 500;
            }


            /* Buttons */
            [data-testid="stSidebar"] .stButton>button {
                width: 100%;
                border-radius: 8px;
                border: none;
                margin-bottom: 8px;
                font-weight: 600;
            }


            /* New Chat button */
            [data-testid="stSidebar"] .stButton>button:first-child {
                background-color: #7B51B5;
                color: white;
            }
            [data-testid="stSidebar"] .stButton>button:first-child:hover {
                background-color: #6a459b;
            }


            /* Process Documents button */
            [data-testid="stSidebar"] .stButton>button[kind="primary"] {
                background-color: #7B51B5 !important;
                color: #5302cc !important;
            }


            /* Browse files button */
            div[data-testid="stFileUploader"] section div div button {
                background-color: #7B51B5 !important;
                color: #FFFFFF !important;
                border-radius: 8px !important;
                font-weight: 600 !important;
                border: none !important;
                padding: 0.5rem 1rem !important;
            }


            div[data-testid="stFileUploader"] section div div button:hover {
                background-color: #6a459b !important;
                color: #FFFFFF !important;
            }


            /* Chat History Buttons */
            [data-testid="stSidebar"] .stButton>button[kind="secondary"] {
                background-color: #2e2e48;
                border: 1px solid #4a4a6a;
                color: white;
            }
            [data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover {
                background-color: #4a4a6a;
            }


            /* Chat messages */
            .chat-message {
                padding: 1rem;
                border-radius: 0.5rem;
                margin-bottom: 1rem;
                display: flex;
                align-items: flex-start;
                max-width: 80%;
                box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
            }
            /* User message bubble - left */
            .chat-message.user {
            background-color: #E8DEF5;
            color: #1c1b1f;
            padding: 0.8rem 1rem;
            border-radius: 12px;
            max-width: 60%;
            margin: 0.5rem 0;
            display: inline-block;
            text-align: left;
            word-wrap: break-word;
            }


            /* AI message bubble - right */
            .chat-message.bot {
             background-color: #FFFFFF;
             color: #000;
             padding: 1rem 1rem;
            border-radius: 8px;
            max-width: 65%;
            margin: 0.5rem 0;
            display: inline-block;
            float: right;
            clear: both;
            word-wrap: break-word;
            }
            .chat-message .avatar {
                width: 40px;
                height: 40px;
                border-radius: 50%;
                margin-left: 1rem;
                flex-shrink: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 1.5rem;
            }
            .chat-message.bot .avatar { background-color: #7B51B5; }
            .chat-message.user .avatar { background-color: #B4A0E5; }


            /* Hide Streamlit default header/footer */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
               
            * Follow-up list */
            .followup-list {
            list-style-type: disc;
            margin-left: 20px;
            color: #7B51B5;
            }
            .followup-list button {
             background-color: #7B51B5;
             color: white;
             border: none;
             border-radius: 6px;
             padding: 4px 8px;
             margin: 2px 0;
             cursor: pointer;
            }
            .followup-list button:hover {
            background-color: #6a459b;
            }
           
        </style>
    """, unsafe_allow_html=True)
# Call the function to apply CSS
load_css()
# ----------------- GLOBAL STYLES -----------------
st.markdown("""
<style>
.followup {
    color: #6c757d;
    font-style: italic;
    margin-left: 10px;
}
</style>
""", unsafe_allow_html=True)


# ----------------- RAG CORE LOGIC -----------------
@st.cache_resource
def load_embeddings_model():
    """Loads the embedding model and caches it."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_data
def load_and_split_pdf(pdf_path: str):
    """Loads and splits a single PDF, optimized with caching."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return text_splitter.split_documents(pages)


def create_rag_chain(vectorstore):
    """Creates the RAG chain for question answering."""
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))


    prompt = ChatPromptTemplate.from_template("""
You are an expert assistant. Answer the user's question based ONLY on the following context.


If the answer is not in the context, say "I don't have enough information from the documents to answer that."


Provide a detailed and well-structured answer.


<context>
{context}
</context>


Question: {input}
""")


    retriever = vectorstore.as_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    return rag_chain




               
# ----------------- SESSION STATE INITIALIZATION -----------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = False
if "uploaded_files_persist" not in st.session_state:
    st.session_state.uploaded_files_persist = []
if "chat_id" not in st.session_state:
    st.session_state.chat_id = f"chat_{int(time.time())}"
if "followups" not in st.session_state:
    st.session_state.followups = []




st.session_state.messages = load_messages(st.session_state.chat_id) if st.session_state.chat_id else []


# ----------------- SIDEBAR UI -----------------
with st.sidebar:
    st.header("⚙️ RAG Configuration")


    # --- NEW CHAT BUTTON ---
    if st.button("➕ New Chat"):
        st.session_state.chat_id = f"chat_{int(time.time())}"
        st.session_state.messages = []
        st.session_state.rag_chain = None
        st.session_state.vectorstore_ready = False
        st.rerun()


    # --- SAVED CHAT SESSIONS ---
    sessions = get_chat_sessions()
    if sessions:
        st.markdown("### 💬 Saved Chats")
        for s in sessions:
            if st.button(f"- {s}", key=f"chat_{s}"):
                st.session_state.chat_id = s
                st.session_state.messages = load_messages(s)
                st.rerun()


    # --- DOCUMENT UPLOAD ---
    st.subheader("📁 Documents")
    uploaded_files = st.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_files_persist = [f.name for f in uploaded_files]


        if st.button("⚙️ Process Documents"):
            with st.spinner("Processing documents..."):
                all_docs = []
                temp_dir = tempfile.mkdtemp()
                for uploaded_file in uploaded_files:
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getvalue())
                    docs = load_and_split_pdf(temp_path)
                    all_docs.extend(docs)


                if all_docs:
                    embeddings = load_embeddings_model()
                    vectorstore = FAISS.from_documents(all_docs, embeddings)
                    st.session_state.rag_chain = create_rag_chain(vectorstore)
                    st.session_state.vectorstore_ready = True
                    st.success("✅ Documents processed successfully!")
                else:
                    st.warning("⚠️ No documents were found to process.")


    # --- DISPLAY UPLOADED DOCS ---
    if st.session_state.uploaded_files_persist:
        st.markdown("### 📄 Uploaded Documents")
        for doc in st.session_state.uploaded_files_persist:
            st.markdown(f"- {doc}")


    st.divider()


    # --- CLEAR CHAT HISTORY ---
    if st.button("🗑️ Clear Current Chat History"):
        clear_chat(st.session_state.chat_id)
        st.session_state.messages = []
        st.success("Chat history cleared!")
        st.rerun()


# ----------------- MAIN CHAT INTERFACE -----------------
st.title("📄 RAG Chatbot")
st.caption("Powered by LangChain, Groq, and Streamlit")


# --- DISPLAY CHAT MESSAGES ---
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


    # Follow-up suggestion under assistant replies
    if message["role"] == "assistant" and i > 0:
        suggestion_topic = st.session_state.messages[i-1]["content"].split()[0]
        st.markdown(f"<div class='followup'>💡 Ask more about: *{suggestion_topic}*</div>", unsafe_allow_html=True)


# ----------------- AI RESPONSE + FOLLOW-UP FUNCTION -----------------
def get_rag_response_and_followups(query):
    """Get AI answer and follow-up questions."""
    # Stream AI answer fully
    full_answer = ""
    for chunk in st.session_state.rag_chain.stream({"input": query}):
        if "answer" in chunk:
            full_answer += chunk["answer"]


    # Generate follow-up questions after answer is complete
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))


    followup_prompt = f"""
Based on the following user question and assistant response, suggest 2-3 relevant follow-up questions that the user might ask next.
Return the questions as a JSON list.


User question: {query}
Assistant answer: {full_answer}
"""
    followups = llm.predict(followup_prompt)
    try:
        import json
        followup_questions = json.loads(followups)
        followup_questions = followup_questions[:2]  # ensure only 2 questions
    except:
        followup_questions = []
   
    # Store follow-ups in session_state to persist after rerun
    st.session_state.followups = followup_questions


    return full_answer, followup_questions


# ----------------- CHAT INPUT -----------------
if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(st.session_state.chat_id, "user", prompt)


    with st.chat_message("user"):
        st.markdown(prompt)


    if not st.session_state.vectorstore_ready:
        st.warning("Please upload and process your documents before asking questions.")
        st.stop()


    with st.chat_message("assistant"):
        # Get AI answer + follow-up questions
        full_response, followups = get_rag_response_and_followups(prompt)
        st.markdown(full_response)


    # Save AI response
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    save_message(st.session_state.chat_id, "assistant", full_response)


    # Save follow-ups persistently
    st.session_state.followups = followups


    ## Display follow-ups in neat list with bubbles
    if st.session_state.followups:
        st.markdown("<ul class='followup-list'>", unsafe_allow_html=True)
        # Use a copy to iterate to avoid issues if we modify the original list
    for i,fq in enumerate(st.session_state.followups.copy()):
        # Button to auto-send follow-up
        if st.button(fq, key=f"followup_{i}"):
            st.session_state.messages.append({"role":"user", "content": fq})
            save_message(st.session_state.chat_id, "user", fq)
            # Clear follow-ups so they don't duplicate on rerun
            st.session_state.followups = []
            st.experimental_rerun()
        else:
            st.markdown(f"<li>{fq}</li>", unsafe_allow_html=True)  # List item
    st.markdown("</ul>", unsafe_allow_html=True)  # Close list
#       st.rerun() # Rerun to process follow-up



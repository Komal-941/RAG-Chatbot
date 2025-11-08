# 🤖 RAG Chatbot – Chat Privately with Your PDFs 🔐  
*(PrivAI Docs | Generative AI Internship Project @ e-Zest Solutions)*  

An intelligent **Retrieval-Augmented Generation (RAG)** Chatbot built using **LangChain**, **Groq API**, **HuggingFace Embeddings**, and **Streamlit**.  
It allows you to **upload PDFs and chat securely** with your data in real-time.  


---

## 📽️ Demo Video  
🎥 [Watch Demo Here](https://drive.google.com/file/d/1XufYBo-QcDpRk-eItXikut4MkWEDTlDl/view?usp=sharing)  

---

## 🧠 Project Overview  
This chatbot combines **LLMs** with **vector-based document retrieval** to answer queries contextually using your uploaded PDFs.  
It ensures **data privacy** — everything runs locally, no external data sharing.  

🔹 **Type:** Generative AI / NLP Project  
🔹 **Tech Stack:** Python, LangChain, Streamlit, Groq API, FAISS, HuggingFace Embeddings  
🔹 **Goal:** Enable private, context-aware conversations from your own documents 

---

## ✨ Features  
- 📄 **Privacy-Preserving Chat** – Query PDFs locally without exposing data online
- 🧠 **Multi-PDF Uploads** – Chat with multiple documents in one session 
- 🔍 **Contextual Retrieval** – Uses **FAISS**-based **vector embeddings** for efficient search  
- 💬 **Smart Prompting** – Answers strictly from your document context  
- ⚙️ **Cache-Optimized Loading** – Faster PDF processing with caching
- 💾 **Persistent Chat History** – Stored using **SQL** for each session  
- 🌐 **Clean UI** – Interactive chatbot interface powered by **Streamlit**

---

## 🧠 Workflow  

- 1️⃣ Upload one or more PDFs
- 2️⃣ PDF content is loaded → split → chunked → vector embeddings created
- 3️⃣ FAISS Vector Store stores document embeddings
- 4️⃣ User queries are passed through RAG chain
- 5️⃣ LLM (ChatGroq) responds using only context from uploaded documents
- 6️⃣ If info not found → “I don’t have enough information from the documents to answer that.”


---

## 🧩 Tech Stack  
- **Backend**	    -  Python, Flask
- **Frontend**	  -  StreamlitEmbeddings	HuggingFaceEmbeddings (all-MiniLM-L6-v2)
- **Vector-Store** -  FAISS
- **LLM Model**   -  ChatGroq (openai/gpt-oss-120b)
- **PDF Parsing**-  PyPDFLoader
- **Database**	 -   SQL (for chat session management)
- **Framework** -  	LangChain

---

## 🏗️ File Structure  
```bash
RAG-Chatbot/
│
├── App.py               # Streamlit front-end app
├── database.py          # Handles embeddings, vector store, and retrieval logic
├── requirements.txt     # Required dependencies
└── README.md            # Documentation
```

---

## 🛠️ Setup Instructions  

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/YourUsername/RAG-Chatbot.git
cd RAG-Chatbot
```

---

### 2️⃣ Create a Virtual Environment  
```bash
python -m venv venv
venv\Scripts\activate    # for Windows
source venv/bin/activate # for Mac/Linux
```

---

### 3️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```
💡 *If you don’t have a `requirements.txt` yet:*  
```bash
pip freeze > requirements.txt
```


---

### 4️⃣ Run the Application  
```bash
python app.py
```
---

### 🌐 5️⃣ Open in Browser  
Visit 👉 [http://localhost:8501](http://localhost:8501)

---

### 🎯 Done!  
Your **RAG Chatbot** is now live — chat with your private PDFs safely and instantly 🤖📄 

---
## 💬 Example Use Cases  
✅ Summarize lengthy reports or research papers  
✅ Extract insights or data from business documents  
✅ Get quick answers without reading entire PDFs  
✅ Ideal for students, analysts, and professionals  


---

## 🧑‍💻 Author – Connect with Me  

**👩‍💻 Komal S. Shelar**  
📍 Pune, India  
🔗 [LinkedIn](https://www.linkedin.com/in/komal-s-shelar/) | [GitHub](https://github.com/Komal-941)  
📧 komalshelar941@gmail.com  

---

## 🏷️ Hashtags for Discoverability  
`#RAGChatbot` `#LangChain` `#Groq` `#Streamlit` `#AIProject` `#Python` `#GenerativeAI` `#DataPrivacy` `#HuggingFace` `#LLM` `#Chatbot` `#OpenSource`

---

⭐ **If you found this helpful, give it a star!**  
Your support helps make open-source projects like this grow 🚀

import os
import pickle
import time
import streamlit as st
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Load environment variables (for Hugging Face API token)
load_dotenv()

st.title("RockyBot: News Research Tool 📈")

st.sidebar.title("News Article URLs")

# User Input for URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"
main_placeholder = st.empty()

# 1️⃣ Function to Extract Cleaned Text from Web Pages
def fetch_clean_text(url):
    """Scrape and clean web page content."""
    response = requests.get(url)
    if response.status_code != 200:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove scripts, styles, and navigation junk
    for tag in soup(["script", "style", "meta", "noscript", "header", "footer", "aside"]):
        tag.extract()

    # Extract main content
    paragraphs = soup.find_all("p")
    text = " ".join([p.get_text() for p in paragraphs])

    # Clean text (remove extra spaces and junk)
    text = " ".join(text.split()).strip()
    return text

if process_url_clicked:
    # 2️⃣ Load and Clean Data
    main_placeholder.text("🔄 Loading and Cleaning Data...")
    docs = [Document(page_content=fetch_clean_text(url), metadata={"source": url}) for url in urls if fetch_clean_text(url)]
    
    if not docs:
        main_placeholder.text("⚠️ No valid content extracted from URLs.")
        st.stop()

    # 3️⃣ Split Documents
    main_placeholder.text("🔄 Splitting Text into Chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    split_docs = text_splitter.split_documents(docs)

    # 4️⃣ Generate Embeddings (Using Hugging Face Model)
    main_placeholder.text("🔄 Generating Embeddings...")
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5️⃣ Build FAISS Vector Store
    vectorstore = FAISS.from_documents(split_docs, embedding_model)
    main_placeholder.text("✅ FAISS Vector Store Created!")

    # 6️⃣ Save FAISS Index to Disk
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)
    
    st.success("✅ Processing complete! Ask a question below.")

# 7️⃣ Load Hugging Face LLM (FLAN-T5 for text generation)
llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5})

# 8️⃣ Question Input and Answer Retrieval
query = main_placeholder.text_input("💬 Ask a Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        # Create the QA Chain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        # Get answer
        result = chain({"question": query}, return_only_outputs=True)

        # 9️⃣ Display Answer
        st.header("🤖 Answer")
        st.write(result["answer"])

        # 1️⃣0️⃣ Display Sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("📌 Sources:")
            sources_list = sources.split("\n")  # Split sources by newline
            for source in sources_list:
                st.write(source)




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

# st.title("RockyBot: News Research Tool 📈")

# st.sidebar.title("News Article URLs")

# # User Input for URLs
# urls = []
# for i in range(3):
#     url = st.sidebar.text_input(f"URL {i+1}")
#     urls.append(url.strip())

# process_url_clicked = st.sidebar.button("Process URLs")
# file_path = "faiss_store.pkl"
# main_placeholder = st.empty()

# # 1️⃣ Function to Extract Cleaned Text from Web Pages
# def fetch_clean_text(url):
#     """Scrape and clean web page content."""
#     response = requests.get(url)
#     if response.status_code != 200:
#         return None

#     soup = BeautifulSoup(response.text, "html.parser")

#     # Remove scripts, styles, and navigation junk
#     for tag in soup(["script", "style", "meta", "noscript", "header", "footer", "aside"]):
#         tag.extract()

#     # Extract main content
#     paragraphs = soup.find_all("p")
#     text = " ".join([p.get_text() for p in paragraphs])

#     # Clean text (remove extra spaces and junk)
#     text = " ".join(text.split()).strip()
#     return text

# if process_url_clicked:
#     # 2️⃣ Load and Clean Data
#     main_placeholder.text("🔄 Loading and Cleaning Data...")
#     docs = [Document(page_content=fetch_clean_text(url), metadata={"source": url}) for url in urls if fetch_clean_text(url)]
    
#     if not docs:
#         main_placeholder.text("⚠️ No valid content extracted from URLs.")
#         st.stop()

#     # 3️⃣ Split Documents
#     main_placeholder.text("🔄 Splitting Text into Chunks...")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
#     split_docs = text_splitter.split_documents(docs)

#     # 4️⃣ Generate Embeddings (Using Hugging Face Model)
#     main_placeholder.text("🔄 Generating Embeddings...")
#     embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#     # 5️⃣ Build FAISS Vector Store
#     vectorstore = FAISS.from_documents(split_docs, embedding_model)
#     main_placeholder.text("✅ FAISS Vector Store Created!")

#     # 6️⃣ Save FAISS Index to Disk
#     with open(file_path, "wb") as f:
#         pickle.dump(vectorstore, f)
    
#     st.success("✅ Processing complete! Ask a question below.")

# # 7️⃣ Load Hugging Face LLM (FLAN-T5 for text generation)
# llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature": 0.5})

# # 8️⃣ Question Input and Answer Retrieval
# query = main_placeholder.text_input("💬 Ask a Question:")
# if query:
#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vectorstore = pickle.load(f)

#         # Create the QA Chain
#         chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

#         # Get answer
#         result = chain({"question": query}, return_only_outputs=True)

#         # 9️⃣ Display Answer
#         st.header("🤖 Answer")
#         st.write(result["answer"])

#         # 1️⃣0️⃣ Display Sources
#         sources = result.get("sources", "")
#         if sources:
#             st.subheader("📌 Sources:")
#             sources_list = sources.split("\n")  # Split sources by newline
#             for source in sources_list:
#                 st.write(source)


import os
import pickle
import streamlit as st
import requests
import faiss
import time
import re
from newspaper import Article
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.llms import LlamaCpp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Streamlit UI
st.set_page_config(page_title="News Research AI", page_icon="📈")
st.title("📰 News Research AI")
st.sidebar.title("📎 Input News URLs")

# Store user-inputted URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("🔄 Process URLs")
file_path = "faiss_index.pkl"

# Function to fetch and clean article text
def fetch_clean_text(url):
    """Fetches and cleans an article from a given URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces/newlines
        return text
    except Exception as e:
        return None

if process_url_clicked:
    st.write("🔄 **Processing URLs...**")

    # Load and clean articles
    docs = []
    for url in urls:
        if url.strip():
            article_text = fetch_clean_text(url)
            if article_text:
                docs.append(Document(page_content=article_text, metadata={"source": url}))

    if not docs:
        st.error("❌ No valid content found! Check your URLs and try again.")
    else:
        # Split documents into smaller chunks for retrieval
        st.write("✂️ **Splitting documents into chunks...**")
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = splitter.split_documents(docs)

        # Generate embeddings & save FAISS index
        st.write("🔢 **Creating embeddings and saving FAISS index...**")
        embedding_model = HuggingFaceEmbeddings(model_name="hkunlp/instructor-xl")  # Stronger than MiniLM
        vectorstore = FAISS.from_documents(split_docs, embedding_model)

        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("✅ **Vector database successfully created!**")

# Initialize Local LLM (Llama 2-7B)
llm_model_path = "models/llama-2-7b-chat.ggmlv3.q4_K_M.bin"  # Path to local model
if not os.path.exists(llm_model_path):
    st.error("❌ Llama model not found! Download it from Meta and place it in 'models/' folder.")

llm = LlamaCpp(
    model_path=llm_model_path,
    temperature=0.7,
    max_tokens=500
)

# User input for questions
query = st.text_input("🔍 Ask a question about the articles:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        st.write("🤖 **Generating response...**")
        result = chain({"question": query}, return_only_outputs=True)

        st.header("💡 Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("📌 Sources:")
            for source in sources.split("\n"):
                st.write(source)
    else:
        st.error("❌ No FAISS index found. Click 'Process URLs' first!")


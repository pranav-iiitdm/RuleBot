import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import requests
from pypdf import PdfReader
from io import BytesIO
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
import json
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain_community.document_loaders import WebBaseLoader
import certifi
from urllib.parse import urljoin
from collections import deque
from requests.exceptions import SSLError

st.title("RuleBot")

st.sidebar.header("Choose a Sport")
sport = st.sidebar.selectbox("Select a sport", ["Formula1", "Cricket"])

if sport == "Formula1":
    st.sidebar.write("You selected Formula1.")
elif sport == "Cricket":
    st.sidebar.write("You selected Cricket.")

# Paths to save/load data
vector_store_dir = "faiss_vector_store"
processed_data = "processed_urls.json"

if os.path.exists(processed_data):
    with open(processed_data, "r") as f:
        processed_urls = json.load(f)
else:
    processed_urls = []

from data import pdf_urls, web_urls
all_urls = pdf_urls + web_urls
new_urls = [url for url in all_urls if url not in processed_urls]
print('49', new_urls)

# Function to load documents from PDF URLs
def load_pdf_documents(_urls):
    docs = []
    for url in _urls:
        response = requests.get(url, verify=certifi.where())
        pdf_file = BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page].extract_text()
        docs.append(Document(page_content=text, metadata={"source": url}))
    return docs

# Function to load documents from websites using webbaseloader
# def load_web_documents(_urls, max_depth=2):
#     docs = []
#     web_loader = WebBaseLoader(_urls)
#     docs = web_loader.load()

#     # for url in _urls:
#     #     try:
#     #         html_content = web_loader.load(url)
#     #         soup = BeautifulSoup(html_content, 'html.parser')
#     #         text = soup.get_text(separator="\n", strip=True)
#     #         docs.append(Document(page_content=text, metadata={"source": url}))
#     #     except Exception as e:
#     #         print(f"Failed to fetch or process {url}: {e}")

#     return docs

# Load new documents only if there are new URLs
new_pdf_docs = load_pdf_documents([url for url in new_urls if url.endswith('.pdf')])
# new_web_docs = load_web_documents([url for url in new_urls if not url.endswith('.pdf')], max_depth=2)

def custom_web_scraper(url):
    try:
        response = requests.get(url, verify=certifi.where())
        response.raise_for_status()
    except SSLError as e:
        print(f"SSL Error occurred while accessing {url}: {e}")
        return None
    except requests.RequestException as e:
        print(f"Error occurred while accessing {url}: {e}")
        return None

    soup = BeautifulSoup(response.content, 'html.parser')
    
    main_content = soup.select_one('main')
    
    if main_content:
        text = main_content.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)
    
    return Document(page_content=text, metadata={"source": url})

new_web_docs = [custom_web_scraper(url) for url in web_urls]

def recursive_scraper(start_url, max_depth=2):
    visited = set()
    queue = deque([(start_url, 0)])
    docs = []

    while queue:
        url, depth = queue.popleft()
        if url in visited or depth > max_depth:
            continue

        visited.add(url)
        doc = custom_web_scraper(url)
        if doc:
            docs.append(doc)

        if depth < max_depth:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            for link in soup.find_all('a', href=True):
                new_url = urljoin(url, link['href'])
                if new_url.startswith(start_url):  # Stay on the same domain
                    queue.append((new_url, depth + 1))

    return docs

new_web_docs = []
for url in web_urls:
    new_web_docs.extend(recursive_scraper(url))

print('83', new_web_docs)

# Combine all new documents
final_documents = new_pdf_docs + new_web_docs

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
final_documents = text_splitter.split_documents(final_documents)

# Initialize embeddings
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# Load existing vector store or create a new one if it doesn't exist
if os.path.exists(vector_store_dir):
    vector_store = FAISS.load_local(vector_store_dir, embeddings=huggingface_embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = None

# update the vector store if there are new documents
if final_documents:
    if vector_store is None:
        vector_store = FAISS.from_documents(final_documents, huggingface_embeddings)
    else:
        vector_store.add_documents(final_documents)

    vector_store.save_local(vector_store_dir)

    processed_urls.extend(new_urls)
    with open(processed_data, "w") as f:
        json.dump(processed_urls, f)

# Initialize the language model
llm = ChatGroq(
    groq_api_key='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',
    model_name="Llama3-8b-8192"
)

# Prompt template creation
prompt = ChatPromptTemplate.from_template(
    """Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}"""
)

# Document chain creation
document_chain = create_stuff_documents_chain(llm, prompt)

# Set up retriever
retriever = vector_store.as_retriever()

# Create retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to process user input
def process_query():
    user_query = st.session_state.user_query_input

    conversation_context = "\n".join(
        [f"User: {msg['user']}\nBot: {msg['bot']}" for msg in st.session_state.chat_history]
    )

    response = retrieval_chain.invoke({
        'input': user_query,
        'context': conversation_context
    })

    st.session_state.chat_history.append({
        'user': user_query,
        'bot': response['answer']
    })

    st.session_state.user_query_input = ''

# Display the chat history
st.subheader("Conversation")
for i, chat in enumerate(st.session_state.chat_history):
    st.write(f"**User**: {chat['user']}")
    st.write(f"**Bot**: {chat['bot']}")

# Interactive query input using a placeholder and callback
user_query_input = st.text_input(
    "Enter your question:",
    key="user_query_input",
    on_change=process_query
)

# Adding a button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state.chat_history = []

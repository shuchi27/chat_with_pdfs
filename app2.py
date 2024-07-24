import streamlit as st
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import concurrent.futures
import openai
import numpy as np
import faiss
import os
import re
import json
import pandas as pd
import logging
import time
import tiktoken
import pickle

logging.basicConfig(level=logging.INFO)

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large")

# Create embeddings and store in FAISS index with metadata
def create_and_store_embeddings(chunks, index_dir="db2"):
    embeddings_list = []
    documents = []

    for chunk in chunks:
        if isinstance(chunk, Document):
            text_content = chunk.page_content  # Assuming 'page_content' is the attribute holding the text
            documents.append(chunk)  # Store the document
        else:
            text_content = chunk  # Assuming chunk is a string if it's not a Document object

        # Generate embedding for the text content and append to list
        embeddings_list.append(embedding.embed_query(text_content))
    
    # Stack embeddings vertically to create a 2D array
    embeddings = np.vstack(embeddings_list)
    
    dimension = embeddings.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Ensure the directory exists
    os.makedirs(index_dir, exist_ok=True)

    # Save index and documents
    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))
    with open(os.path.join(index_dir, "index.pkl"), 'wb') as f:
        pickle.dump(documents, f)

    return index

# Load or create the vector store

def load_vectorstore(text_chunks=None, index_dir="db2"):
    index_path = os.path.join(index_dir, "index.faiss")
    pkl_path = os.path.join(index_dir, "index.pkl")

    if os.path.exists(index_path) and os.path.exists(pkl_path):
        index = faiss.read_index(index_path)
        with open(pkl_path, 'rb') as f:
            documents = pickle.load(f)
        if text_chunks:
            new_index = create_and_store_embeddings(text_chunks, index_dir=index_dir)
            index = merge_faiss_indices(index, new_index)
            faiss.write_index(index, index_path)
            with open(pkl_path, 'wb') as f:
                pickle.dump(documents + text_chunks, f)
        return index, documents
    else:
        if text_chunks:
            index = create_and_store_embeddings(text_chunks, index_dir=index_dir)
            faiss.write_index(index, index_path)
            with open(pkl_path, 'wb') as f:
                pickle.dump(text_chunks, f)
            return index, text_chunks
    return None, None

# Function to create a custom FAISS index
def create_custom_faiss_index(dimension, nlist, m, bits_per_subvector):
    quantizer = faiss.IndexFlatL2(dimension)  # Using L2 distance for the coarse quantizer
    index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, bits_per_subvector)
    index.own_fields = True
    index.make_direct_map()  # Initialize the direct map
    return index

# Merge FAISS indices
def merge_faiss_indices(index1, index2):
    vectors1 = [index1.reconstruct(i) for i in range(index1.ntotal)]
    vectors2 = [index2.reconstruct(i) for i in range(index2.ntotal)]
    combined_vectors = np.vstack((vectors1, vectors2))
    new_index = create_custom_faiss_index(index1.d)
    new_index.train(combined_vectors)
    new_index.add(combined_vectors)
    return new_index

def isFilesExist(uploaded_file_name, unique_docs_list):
    st.write("-----------------")

    existing_files = [file_name for file_name in uploaded_file_name if file_name in unique_docs_list]
    if existing_files:
        st.write("------true-------")

        st.error(f"Error: {', '.join(existing_files)} already exist in the vector store.")
        return False
    else:
        st.write("------false-------")

        unique_docs_list.extend(uploaded_file_name)
        return True

def get_pdf_content(files):
    pattern =  r"^\d+(?:\.\d+)?\s+(.*)"
    chunkList = []
    for file in files:
        pdf_reader = PdfReader(file)
        documents = []
        page_number=0
        for page in pdf_reader.pages:
            text = page.extract_text()  
            matches = re.findall(pattern, text, re.MULTILINE)
            titles = matches if len(matches) != 0 else []
            page_number += 1
            metadata = {
                'source': file.name,
                'page': page_number,
                'heading': titles,
            }
            pdf_data = {
                "page_content": text,
                "metadata": metadata
            }
            document = Document(json.dumps(pdf_data))
            documents.append(document)

        text_chunks = get_text_chunks(documents)
        chunkList.extend(text_chunks)
    return chunkList

def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_documents(pages)
    return chunks 

# Custom search function with metadata filtering
def search_with_metadata(query, index, metadata, top_k=5):
    query_embedding = embedding.embed_query([query])
    D, I = index.search(query_embedding, top_k)

    alpha = 0.1
    flat_results = [(index.docstore.search(vector_ids[idx]), np.exp(-alpha * dist)) for dist, idx in zip(D[0], I[0])]
    return flat_results

def model_query(user_question, metadata, num_sources, faiss_indices, selected_file, db_index, index_type):
    st.write("db_index:", db_index)
    flat_results = search_with_metadata(user_question, db_index, metadata, top_k=5)

    retriever = db_index.as_retriever()

    template = """
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. Excerpts from relevant documents the AI has read are included in the conversation and are used to answer questions more accurately. The AI is not perfect, and sometimes it says things that are inconsistent with what it has said before. The AI always replies succinctly with the answer to the question and provides more information when asked. The AI recognizes questions asked to it are usually in reference to the provided context, even if the context is sometimes hard to understand, and answers with information relevant from the context.
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(input_variables=["context", "history", "question"], template=template)

    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(memory_key="history", input_key="question")

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={"verbose": True, "prompt": prompt, "memory": st.session_state.conversation_memory},
    )

    ai_message = qa.run(user_question)

    return ai_message, flat_results

def extract_vectors_from_db(db_index):
    vectors, ids = [], []
    for idx in range(db_index.ntotal):
        vectors.append(db_index.reconstruct(int(idx)))
        ids.append(idx)
    return vectors, ids

def getDocNamesFromIndexFile(index_dir="db2"):
    # Define the path to the index pickle file where documents are stored
    index_path = os.path.join(index_dir, "index.pkl")
    
    # Check if the index file exists
    if not os.path.exists(index_path):
        print(f"No index file found in {index_dir}")
        return []

    # Load the documents from the pickle file
    with open(index_path, 'rb') as f:
        documents = pickle.load(f)

    # Assuming documents are stored as instances of a class or dicts with metadata
    # Extract unique document names from the metadata in these documents
    unique_docs = set()
    for doc in documents:
        # Check if the document has a 'metadata' attribute or key and 'source' within that
        if isinstance(doc, dict):
            # Assuming the document is stored as a dictionary
            if 'metadata' in doc and 'source' in doc['metadata']:
                unique_docs.add(doc['metadata']['source'])
        elif hasattr(doc, 'metadata') and 'source' in doc.metadata:
            # Assuming the document is an instance of a class with a 'metadata' attribute
            unique_docs.add(doc.metadata['source'])

    return list(unique_docs)



# Function to convert FAISS index to a DataFrame
def faiss_index_to_dataframe(db_index):
    vectors, ids = extract_vectors_from_db(db_index)
    # Convert to DataFrame
    df = pd.DataFrame(vectors)
    df['ID'] = ids
    return df

def main():
    db_index = None
    doc_list = []
    if os.path.exists("db2"):
        vector_start_time = time.time()
        db_index,metadata = load_vectorstore(index_dir="db2")
        df = faiss_index_to_dataframe(db_index)
        vector_end_time = time.time()
        vector_load_time = vector_end_time - vector_start_time
        print(f"Time taken to load vector: {vector_load_time} seconds")
        doc_list = getDocNamesFromVectorStore(db_index)


    if 'title' not in st.session_state:
        st.session_state.title = "ChatGPT with Document Query"  # Default title

    if 'model' not in st.session_state:
        st.session_state.model = None  # Default title
        
    if 'is_summary' not in st.session_state:
        st.session_state.is_summary = False  

    if 'summary' not in st.session_state:
        st.session_state.summary = None  # Default summary

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "is_vectorstore_update" not in st.session_state:
        st.session_state.is_vectorstore_update = False
    

    #st.set_page_config("ChatGPT with Document Query", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    vectorStore = None
    faiss_indices = None

    with st.sidebar:
        st.title("Document Query Settings")
        uploaded_file_names = []
        files = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
        text_chunks = ""

        if st.button("Load Data"):
            with st.spinner("Loading Data..."):
                st.write("--------1--------")

                if files:
                    st.write("--------2--------")
                    for f in files:
                        uploaded_file_names.append(f.name)

                if doc_list:
                    st.write("--------3--------")
                    isFile = isFilesExist(uploaded_file_names, doc_list)
                    st.write(isFile)

                chunkList = get_pdf_content(files)
                db_index,metadata = load_vectorstore(chunkList, index_dir="db2")
                vectors, vector_ids = extract_vectors_from_db(db_index)

        num_sources = st.slider("Number of sources per document:", min_value=1, max_value=5, value=3)
        index_type = st.selectbox("Choose faiss index to search:", options=["L2", "ivf", "ivfpq", "hsnw", "similarity_search"], index=0)
        model_version = st.selectbox("Select the GPT model version:", options=["gpt-4o", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"], index=0)
        st.session_state.model = model_version
        llm = ChatOpenAI(temperature=0, model=st.session_state.model)
        select_option = st.selectbox("Select option to query:", options=["Entire Database", "Selected File"], index=0)
        st.write(db_index)
        sorted_doc_list = sorted(doc_list)
        selected_file = None
        
        if select_option == "Entire Database":
            st.write("You selected to query the entire database.")
            with st.expander("Available Documents"):
                if doc_list:
                    st.write(sorted_doc_list)

    st.title("ChatGPT with Document Query")
    chat_container = st.container()
    with chat_container:
        response_text = st.empty()

    user_question = st.chat_input("Ask something", key="prompt_input")

    if user_question:
        st.session_state.is_summary = False
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        query_start_time = time.time()
        model_response, context, final_time, embedding_time = model_query(user_question,metadata, num_sources, faiss_indices, selected_file, db_index, index_type)
        query_end_time = time.time()
        query_load_time = query_end_time - query_start_time
        print(f"Time taken to search: {final_time} ms")
        print(f"Time taken for response: {query_load_time} sec")
        print(f"Time taken for embedding: {embedding_time:.2f} ms")

        st.session_state.chat_history.append({"role": "model", "content": model_response})
        if context:
            st.session_state.chat_history.append({"role": "context", "content": context})

    chat_container = st.container()
    with chat_container:
        with st.container():
            st.markdown('<div class="chat-box">', unsafe_allow_html=True)
            if st.session_state.is_summary:
                st.write(st.session_state.summary)
            else:
                if user_question:
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            st.markdown(f"> **User**: {message['content']}")
                        elif message["role"] == "model":
                            st.markdown(f"> **Model**: {message['content']}")
                        elif message["role"] == "context":
                            context = message["content"]
                            with st.expander("Click to see the context"):
                                for doc, relevance in context:
                                    st.markdown(f"> **Context Document**: {doc.metadata['source']}")
                                    st.markdown(f"> **Page Number**: {doc.metadata['page']}")
                                    st.markdown(f"> **Content**: {doc.page_content}")
                                    st.markdown(f"> **Relevancy**: {relevance}")
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
            <style>
                .reportview-container .main .block-container {
                    padding-top: 0rem;
                    padding-bottom: 5rem;
                }
                .footer {
                    position: fixed;
                    bottom: 0;
                    width: 100%;
                    background-color: #f1f1f1;
                    padding: 10px;
                    text-align: center;
                }
            </style>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()

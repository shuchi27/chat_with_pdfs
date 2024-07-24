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
from ragas import evaluate
from datasets import Dataset
from ragas.metrics.critique import harmfulness
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_similarity, answer_correctness
import concurrent.futures
import openai
import numpy as np
import faiss
import os
import re
import json
import pandas as pd
import logging
import pickle
logging.basicConfig(level=logging.INFO)
import tiktoken
import uuid
from langchain.schema import BaseRetriever, Document
from typing import List
load_dotenv()
from pydantic import BaseModel, Field
import time
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(temperature=0)


def save_vectors_and_chunks(index, docstore, index_to_docstore_id, chunks, faiss_index_file, pkl_file):
    if not os.path.exists("db2"):
        os.makedirs("db2")
    faiss.write_index(index, faiss_index_file)
    with open(pkl_file, 'wb') as f:
        pickle.dump((docstore, index_to_docstore_id, chunks), f)

def load_vectors_and_chunks(faiss_index_file, pkl_file):
    index = faiss.read_index(faiss_index_file)
    with open(pkl_file, 'rb') as f:
        docstore, index_to_docstore_id, chunks = pickle.load(f)
    return index, docstore, index_to_docstore_id, chunks

def create_and_save_index(text_chunks, faiss_index_file, pkl_file,nlist=None):

    embeddings = embedding.embed_documents(
        [chunk.page_content + " " + " ".join([f"{key}: {value}" for key, value in chunk.metadata.items()])
         for chunk in text_chunks]
    )
    embeddings = embedding.embed_documents([chunk.page_content for chunk in text_chunks])
    dimension = len(embeddings[0])

    num_points = len(embeddings)

    
    # Set nlist to be less than or equal to the number of training points
    if nlist is None or nlist > num_points:
        nlist = min(num_points, 40)  # Default to 40 clusters or less if points are fewer

    # Ensure nprobe is 25% of nlist, but at least 1
    nprobe = max(1, nlist // 4)

    m = 8  # Number of subquantizers, a typical choice

    quantizer = faiss.IndexFlatL2(dimension)
    ivfpq_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 16, 8)  # 8 bits per sub-vector
    ivfpq_index.train(np.array(embeddings).astype(np.float32))
    ivfpq_index.add(np.array(embeddings).astype(np.float32))
    ivfpq_index.nprobe=10
    
    docstore = {str(uuid.uuid4()): chunk for chunk in text_chunks}
    index_to_docstore_id = list(docstore.keys())  # Use UUID keys for index mapping

    save_vectors_and_chunks(ivfpq_index, docstore, index_to_docstore_id, text_chunks, faiss_index_file, pkl_file)
    return ivfpq_index, docstore, index_to_docstore_id

def load_vectorstore(text_chunks=None):
    faiss_index_file = "db2/index.faiss"
    pkl_file = "db2/index.pkl"
    if os.path.exists(faiss_index_file) and os.path.exists(pkl_file):
        index, docstore, index_to_docstore_id, chunks = load_vectors_and_chunks(faiss_index_file, pkl_file)
        db = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embedding.embed_query)
        if text_chunks is not None:
            st.session_state.is_vectorstore_update = True
            new_index, new_docstore, new_index_to_docstore_id = create_and_save_index(text_chunks, faiss_index_file, pkl_file)
            db.index.merge_from(new_index)
            db.docstore.update(new_docstore)
            db.index_to_docstore_id.extend(new_index_to_docstore_id)
            save_vectors_and_chunks(db.index, db.docstore, db.index_to_docstore_id, chunks + text_chunks, faiss_index_file, pkl_file)
    else:
        if text_chunks is not None:
            index, docstore, index_to_docstore_id = create_and_save_index(text_chunks, faiss_index_file, pkl_file)
            db = FAISS(index=index, docstore=docstore, index_to_docstore_id=index_to_docstore_id, embedding_function=embedding.embed_query)
        else:
            raise ValueError("No existing vectorstore found and no text_chunks provided to create a new one.")
    return db

class Document:
    def __init__(self, json_string):
        data = json.loads(json_string)
        self.page_content = data.get("page_content", "")
        self.metadata = data.get("metadata", {})

    def to_json(self):
        data = {
            "page_content": self.page_content,
            "metadata": self.metadata
        }
        return json.dumps(data)

def get_pdf_content(files):
    pattern = r"^\d+(?:\.\d+)?\s+(.*)"
    chunkList = []
    for file in files:
        pdf_reader = PdfReader(file)
        documents = []
        page_number = 0
        for page in pdf_reader.pages:
            text = page.extract_text()
            matches = re.findall(pattern, text, re.MULTILINE)
            titles = matches if len(matches) != 0 else []
            page_number += 1

            metadata = {
                'source': file.name,
                'page': page_number,
                'heading': titles
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
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=350,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

class CustomRetriever(BaseRetriever, BaseModel):
    vector_store: object = Field(...)
    num_results: int = Field(...)

    class Config:
        arbitrary_types_allowed = True

    def get_relevant_documents(self, query: str) -> List[Document]:

        query_vector = self.vector_store.embedding_function(query)
        query_vector = np.array(query_vector).reshape(1, -1)

        # Perform search to get top documents
        D, I = self.vector_store.index.search(query_vector, 5)
        relevant_docs = [self.vector_store.docstore[self.vector_store.index_to_docstore_id[idx]] 
                         for idx in I[0]]
       
        return relevant_docs



def model_query(user_question, num_sources, vectorStore, index_type):
    num_sources=15

    custom_retriever = CustomRetriever(vector_store=vectorStore, num_results=num_sources)

    relevant_docs = custom_retriever.get_relevant_documents(user_question)
    # Extract contexts from relevant documents
    #compressor = FlashrankRerank()
    #compression_retriever = ContextualCompressionRetriever(
        #base_compressor=compressor, base_retriever=custom_retriever
    #)
    # Combine the content of the top documents
   #context = "\n\n".join([doc.page_content for doc in relevant_docs])


    template = """
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. Excerpts from relevant documents the AI has read are included in the conversation and are used to answer questions more accurately. The AI is not perfect, and sometimes it says things that are inconsistent with what it has said before. The AI always replies succinctly with the answer to the question and provides more information when asked. The AI recognizes questions asked to it are usually in reference to the provided context, even if the context is sometimes hard to understand, and answers with information relevant from the context.
    after you get the context, perform following steps.
    step1: read the question
    step2: identify all the important words whichever are required to answer the question
    step3: check if they all exist in the context
    step4: if the context lacks the required information, explain, what is missing and why you cant provide a specific answer. 
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
    prompt = PromptTemplate(
        input_variables=["context", "history", "question"],
        template=template,
    )

    if 'conversation_memory' not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory(
            memory_key="history",
            input_key="question"
        )

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(),
        chain_type='stuff',
        retriever=custom_retriever,        
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": st.session_state.conversation_memory,
        }
    )

    ai_message = qa.run(user_question)
   
    return ai_message,relevant_docs

def getDocNamesFromVectorStore1234(db):
    unique_docs = set()
    v_dict = db.docstore._dict
    for k in v_dict.keys():
        doc_name = v_dict[k].metadata['source']
        unique_docs.add(doc_name)
    unique_docs_list = list(unique_docs)
    return unique_docs_list

def getDocNamesFromVectorStore(db):

    document_names = [chunk.document_name for chunk in db.docstore.values() if hasattr(chunk, 'document_name')]
    return document_names

def isFilesExist(uploaded_file_name, unique_docs_list):
    existing_files = [file_name for file_name in uploaded_file_name if file_name in unique_docs_list]

    if existing_files:
        st.error(f"Error: {', '.join(existing_files)} already exist in the vector store.")
        return False
    else:
        unique_docs_list.extend(uploaded_file_name)
        return True



def extract_vectors_from_db(db):
    vectors = []
    ids = []
    for idx in range(len(db.index_to_docstore_id)):
        vectors.append(db.index.reconstruct(int(idx)))
        ids.append(db.index_to_docstore_id[int(idx)])
    return vectors, ids

def show_vectorstore(db):
    vector_dataframe = store_to_dataframe(db)
    st.write(vector_dataframe)
    
    
def store_to_dataframe(db):
    v_dict  = db.docstore.values()
    data_rows=[]
    st.write("in delete.................")
    for key, value in db.docstore.items():
        doc_name = value.metadata['source']
        page_number = value.metadata['page'] + 1
        content = value.page_content
        data_rows.append({"chunk_id": key, "document": doc_name, "page": page_number, "content": content})
        vector_df = pd.DataFrame(data_rows)
    return vector_df

def main():
    db = None
    doc_list = []
    if os.path.exists("db2"):
        db = load_vectorstore()
        #vectors, vector_ids = extract_vectors_from_db(db)
        doc_list = getDocNamesFromVectorStore(db)

    if 'title' not in st.session_state:
        st.session_state.title = "ChatGPT with Document Query"

    if 'model' not in st.session_state:
        st.session_state.model = None
        
    if 'is_summary' not in st.session_state:
        st.session_state.is_summary = False  

    if 'summary' not in st.session_state:
        st.session_state.summary = None

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "is_vectorstore_update" not in st.session_state:
        st.session_state.is_vectorstore_update = False

    st.set_page_config(st.session_state.title, page_icon=":books:")
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
                if files:
                    st.session_state.uploaded_file = True
                    for f in files:
                        uploaded_file_names.append(f.name) 

                    st.write(uploaded_file_names)

                if doc_list is not None:
                    isFile = isFilesExist(uploaded_file_names, doc_list)

                if isFile:
                    chunkList = get_pdf_content(files)
                    db = load_vectorstore(chunkList)
                    #vectors, vector_ids = extract_vectors_from_db(db)

        files = ""
        num_sources = st.slider("Number of sources per document:", min_value=1, max_value=5, value=3)

        index_type = st.selectbox(
            "Choose faiss index to search:",
            options=["L2", "ivf", "ivfpq", "hsnw", "similarity_search"],
            index=0
        )

        model_version = st.selectbox(
            "Select the GPT model version:",
            options=["gpt-4o", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"],
            index=0
        )

        st.session_state.model = model_version
        llm = ChatOpenAI(temperature=0, model=st.session_state.model)

        sorted_doc_list = sorted(doc_list)

    st.title(st.session_state.title)
    #with st.expander("Show VectorStore"):
            #show_vectorstore(db)

    chat_container = st.container()
    with chat_container:
        response_text = st.empty()

    user_question = st.chat_input("Ask something", key="prompt_input")
    response_start_time=time.time()

    if user_question:
        st.session_state.is_summary = False
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        #model_response, context, final_time = model_query(user_question, num_sources, db, index_type, vectors, vector_ids)

        model_response, context = model_query(user_question, num_sources, db, index_type)
    
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
                                    #for doc in context:
                                    for doc in context:
                                        st.markdown(f"> **Context Document**: {doc.metadata['source']}")
                                        st.markdown(f"> **Page Number**: {doc.metadata['page']}")
                                        st.markdown(f"> **Content**: {doc.page_content}")
                                        #st.markdown(f"> **Relevancy**: {relevance}")
    response_end_time=time.time()
    response_time= (response_end_time-response_start_time)
    print(f"Time taken to respond: {response_time:} sec")
               
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

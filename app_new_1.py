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
from langchain.chains import ConversationalRetrievalChain,RetrievalQA
from htmlTemplates import  css, bot_template, user_template
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
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
import time
import tiktoken
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
import uuid
from langchain.schema import BaseRetriever, Document
from typing import List
from langchain.schema import BaseRetriever, Document
from pydantic import BaseModel, Field

load_dotenv()

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
    #embeddings = embedding.embed_documents([chunk.page_content for chunk in text_chunks])
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
    pattern =  r"^\d+(?:\.\d+)?\s+(.*)"
    chunkList = []
    for file in files:
        pdf_reader = PdfReader(file)
        documents = []
        page_number =0
        for page in pdf_reader.pages:
            text = page.extract_text()  
            matches = re.findall(pattern, text, re.MULTILINE)
            titles = matches if len(matches) != 0 else []
            page_number += 1
            #keywords = extract_keywords(text)  # Extract keywords using the LLM

            metadata = {
                'source': file.name,
                'page': page_number,
                'heading': titles
                #'keywords': keywords  # Add extracted keywords to metadata
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
    is_selected_file: bool = Field(default=False)
    vector_ids: List[int] = Field(default_factory=list)
    

    def get_relevant_documents(self, query: str) -> List[Document]:

        query_vector = embedding.embed_query(query)
        query_vector = np.array(query_vector).reshape(1, -1)

        # Perform search to get top documents
        D, I = self.vector_store.index.search(query_vector,self.num_results)

        if self.is_selected_file:
            # Use vector_ids to fetch documents when a specific file is selected
            vector_ids = extract_vector_ids_from_db(self.vector_store)
            relevant_docs = [(self.vector_store.docstore[vector_ids[int(idx)]])
                             for  idx in I[0]]
        else:
            # Directly use index_to_docstore_id mapping when no specific file is selected
            relevant_docs = [self.vector_store.docstore[self.vector_store.index_to_docstore_id[int(idx)]]
                             for idx in I[0]]
       
        return relevant_docs


def model_query(user_question,num_sources,faiss_indices,selected_file,vectorStore,index_type):
    # Gather related context from documents for each query
     # Load from local storage
    #persisted_vectorstore = load_vectorstore()
    vector_ids = []
    num_sources=15
    is_selected_file = False
    
    if selected_file is not None:
        new_db = faiss_indices[selected_file]
        is_selected_file = True
    else:
        new_db = vectorStore
        st.write(new_db)

    custom_retriever = CustomRetriever(vector_store=new_db, num_results=num_sources,is_selected_file = is_selected_file)

    relevant_docs = custom_retriever.get_relevant_documents(user_question)


    #for doc in docs:
        #logging.info("-------------")
        #logging.info("Score: %s", doc[1])

    #retriever = new_db.as_retriever()
    #compressor = FlashrankRerank()
    #compression_retriever = ContextualCompressionRetriever(
        #base_compressor=compressor, base_retriever=retriever
    #)
    #rileysPrompt()
    #prompt fot 
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
        input_variables=["context","history", "question"],
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
   

def getDocNamesFromVectorStore(db):
    unique_docs = set()
    for key, value in db.docstore.items():
        doc_name = value.metadata['source']
        unique_docs.add(doc_name)
        unique_docs_list = list(unique_docs)
    return unique_docs_list

        
def isFilesExist(uploaded_file_name, unique_docs_list):
    existing_files = [file_name for file_name in uploaded_file_name if file_name in unique_docs_list]
    
    if existing_files:
        st.error(f"Error: {', '.join(existing_files)} already exist in the vector store.")
        return False
    else:
        unique_docs_list.extend(uploaded_file_name)
        return True

         

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


def get_indices_for_file(db, file_name):
    documentList = []
    # Iterate through the existing vectorstore and filter embeddings for the specified file
    for key,value in db.docstore.items():
        doc_name = value.metadata['source']
        
        if doc_name == file_name:
            # Append the embedding of the document to filtered_embeddings
            documentList.append(value)
            #filtered_embeddings.append(db.docstore._dict[key].embedding)
    
    faiss_indices = {
                        str(file_name): FAISS.from_documents(documentList, embedding)
                    }

    # Create a new FAISS vectorstore from the filtered embeddings
    #new_vectorstore = FAISS.from_embeddings(filtered_embeddings)
    return faiss_indices


def hnsw_index_search(query: str, db_ivf, vectors, vector_ids, top_k=5, ef_search=50, ef_construction=200):
    #embedding = OpenAIEmbeddings()
    #embedding = OpenAIEmbeddings(model_name="text-embedding-3-large")
    
    # Step 1: Convert the query to its vector representation
    query_vector = embedding.embed_query(query)
    query_vector = np.array(query_vector).reshape(1, -1)  # Ensure the shape is correct

    dimension = query_vector.shape[1]

    # Create and initialize the HNSW index
    hnsw_index = faiss.IndexHNSWFlat(dimension, 32)  # 32 is the number of neighbors for each node
    
    # Set ef_construction
    hnsw_index.hnsw.efConstruction = ef_construction

    if vectors:
        # Add vectors to the HNSW index
        hnsw_index.add(np.array(vectors, dtype=np.float32))

        # Set the ef parameter for querying (controls the recall)
        hnsw_index.hnsw.efSearch = ef_search  # ef should be greater than or equal to top_k

        # Perform the search
        distances, indices = hnsw_index.search(query_vector, top_k)
        
        # Calculate relevancy scores from distances
        alpha = 0.5
        relevancy_scores = [np.exp(-alpha * d) for d in distances[0]]

        # Retrieve results from docstore based on indices
        hnsw_results = [(db_ivf.docstore.search(vector_ids[i]), relevancy_scores[j]) for j, i in enumerate(indices[0])]
        return hnsw_results

    return []



def extract_vector_ids_from_db(db):
    vector_ids = []
    for idx in range(len(db.index_to_docstore_id)):
        vector_ids.append(db.index_to_docstore_id[int(idx)]) # Get corresponding docstore IDs
    return vector_ids
        

def main():
    db = None
    doc_list=[]
    if os.path.exists("db2"):
        vector_start_time = time.time()
        db = load_vectorstore()
        #vectors,vector_ids = extract_vectors_from_db(db)
        #vector_ids = extract_vector_ids_from_db
        vector_end_time = time.time()
        vector_load_time = vector_end_time - vector_start_time
        print(f"Time taken to load vector: {vector_load_time} seconds")
        doc_list = getDocNamesFromVectorStore(db)

    #if db is not None:
     #show_vectorstore(db)
            

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


    st.set_page_config(st.session_state.title,page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    vectorStore = None
    faiss_indices  = None
    
    with st.sidebar:
        st.title("Document Query Settings")        
        uploaded_file_names = []
        files = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
        text_chunks = ""
        if st.button("Load Data"):
         with st.spinner("Loading Data......"):
            if files:
                st.session_state.uploaded_file = True
                for f in files:
                    uploaded_file_names.append(f.name) 

                st.write(uploaded_file_names)

            if doc_list is not None:
                isFile = isFilesExist(uploaded_file_names, doc_list)

           
            if isFile:
                # get content from uploaded PDFs
                chunkList = get_pdf_content(files)
            
                    
                # split the text chunks
                #text_chunks = get_text_chunks(pages)

                    # st.write("creatig vectorstore.....")
                db = load_vectorstore(chunkList)
                #vector_ids = extract_vector_ids_from_db(db)

                # st.write(vectorStore)    

        files = ""
        # Add a slider for number of sources to return 1-5
        num_sources = st.slider("Number of sources per document:", min_value=1, max_value=5, value=3)

        index_type = st.selectbox(
            "Choose faiss index to search:",
            options=["L2", "ivf", "ivfpq","hsnw","similarity_search"],
            index=0  # Default to gpt-4o
        )

        model_version = st.selectbox(
            "Select the GPT model version:",
            options=["gpt-4o", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"],
            index=0  # Default to gpt-4o
        )


        st.session_state.model = model_version
        llm = ChatOpenAI(temperature=0, model=st.session_state.model)

        
        # Add select box for choosing between Selected File or Entire Database
        select_option = st.selectbox(
            "Select option to query:",
            options=["Entire Database","Selected File"],
            index=0  # Default to Selected File
        )

        sorted_doc_list = sorted(doc_list)
        selected_file = None
        if select_option == "Selected File":
            selected_file = st.selectbox(
                "Please select the file to query:",
                options=sorted_doc_list,
                index=0  # Default to the first file in doc_list
            )

            if selected_file:
                faiss_indices = get_indices_for_file(db,selected_file)
                st.write(faiss_indices)

            #if st.button("delete document"):
                #delete_document(faiss_indices[selected_file],db,selected_file)
                
            if st.button("Summary"):
                st.session_state.is_summary = True
                with st.spinner("Processing..."):

                    if selected_file is None:
                        st.error(f"Error: Please select the file to create summary.")
                    
                    
                    template_string = """
                    Read the text delimited by three angular brackets. \
                    These are the chunks the documents have been split into.
                    Step 1: Summarize Each Chunk
                    Instruction: Please summarize each chunk individually, capturing the main themes and key points.\
                                 Do not make things up; only summarize what is actually present in the text.\
                                 No hallucination.
                    <<<{text}>>>
                    """
                    prompt_template = ChatPromptTemplate.from_template(template_string)

                      # Function to summarize a single chunk
                    def summarize_chunk(chunk):
                        customer_messages = prompt_template.format_messages(text=chunk.page_content)
                        try:
                            summary_response = llm(customer_messages)
                            return summary_response.content
                        except openai.error.OpenAIError as e:
                            print(f"Error: {e}")
                            time.sleep(5)  # Wait before retrying in case of rate limit
                            return None
 
                    if faiss_indices is not None:

                        all_chunks = []
                        newdb = faiss_indices[selected_file]
                        for key in newdb.docstore._dict.keys():
                            doc = newdb.docstore._dict[key]
                            all_chunks.append(doc)

                       # Summarize chunks in parallel
                        start_time = time.time()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            chunk_summaries = list(executor.map(summarize_chunk, all_chunks))

                        # Filter out None values (in case of errors)
                        chunk_summaries = [summary for summary in chunk_summaries if summary]

                        # Final template for creating a summary from chunk summaries
                        final_template_string = """
                        Based on the summaries of all the chunks provided below, \
                        create a final, concise, and to-the-point summary that captures the main themes and key points\
                        from the entire document. Exclude the disclaimer from the summary if any exists in the document.
                        Summaries of chunks:
                        <<<{chunk_summaries}>>>
                        """
                        final_prompt_template = ChatPromptTemplate.from_template(final_template_string)
                        final_customer_messages = final_prompt_template.format_messages(
                            chunk_summaries="\n\n".join(chunk_summaries)
                        )

                        # Generate the final summary
                        final_summary_response = llm(final_customer_messages)

                        end_time = time.time()
                        chain_load_time = end_time - start_time
                        print(f"Time taken: {chain_load_time} seconds")

                        st.session_state.summary = final_summary_response.content

        elif select_option == "Entire Database":
            st.write("You selected to query the entire database.")
            with st.expander("Available Documents"):
                if doc_list:
                    st.write(sorted_doc_list)

        
            
    # Main section
    st.title(st.session_state.title)
   # with st.expander("Show VectorStore"):
            #show_vectorstore(db)


    # Center container for responses
    chat_container = st.container()
    with chat_container:
        response_text = st.empty()

   
   # Lower section for prompt input
    user_question = st.chat_input("Ask something", key="prompt_input")
    # Send button


    if user_question:
        st.session_state.is_summary= False
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        model_response, context = model_query(user_question,num_sources,faiss_indices,selected_file,db,index_type)
        

        st.session_state.chat_history.append({"role": "model", "content": model_response})
        if context:
            st.session_state.chat_history.append({"role": "context", "content": context})
        #handle_userinput()


        # Main chat container
    chat_container = st.container()
    with chat_container:
            with st.container():
                st.markdown('<div class="chat-box">', unsafe_allow_html=True)
                if (st.session_state.is_summary):
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
                                       # st.markdown(f"> **Relevancy**: {relevance}")

            st.markdown('</div>', unsafe_allow_html=True)

    # Use st.rerun() to update the display immediately after sending the message
    #st.rerun()


    # CSS for fixed input area
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


import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from htmlTemplates import  css, bot_template, user_template
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from ragas import evaluate
from datasets import Dataset
#from ragas.metrics.critique import harmfulness
#from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_correctness,answer_similarity
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision, context_recall, context_entity_recall, answer_correctness
import concurrent.futures
import openai
import numpy as np
import faiss
import os
import re
import json
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)
import time
import tiktoken
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank

load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(temperature=0)


def load_vectorstore(text_chunks=None):
    if os.path.exists("db"):
        # Load the existing vectorstore
        db = FAISS.load_local("db", embedding, allow_dangerous_deserialization=True)
        if text_chunks is not None:
            # Create a temporary vectorstore from the new text chunks
            st.session_state.is_vectorstore_update = True

            create_embedding_start_time=time.time()
            new_vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embedding)
            create_embedding_end_time=time.time()
            create_embedding_time = (create_embedding_end_time - create_embedding_start_time) * 1000
            print(f"Time taken to create embeddings: {create_embedding_time:} ms")


            # Merge the new vectorstore into the existing one
            db.merge_from(new_vectorstore)
            # Optionally save the updated vectorstore back to disk
            db.save_local("db")
    else:
        if text_chunks is not None:
            st.session_state.is_vectorstore_update = True

            # Create a new vectorstore from the text chunks
            create_embedding_start_time=time.time()
            db = FAISS.from_documents(documents=text_chunks, embedding=embedding)
            create_embedding_end_time=time.time()
            create_embedding_time = (create_embedding_end_time - create_embedding_start_time) * 1000
            print(f"Time taken to create embeddings: {create_embedding_time:} ms")

            # Save the new vectorstore to disk
            db.save_local("db")
        else:
            # If the directory does not exist and no text_chunks provided, raise an error
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


def model_query(user_question,num_sources,faiss_indices,selected_file,vectorStore,index_type,vectors,vector_ids):
    # Gather related context from documents for each query
     # Load from local storage
    #persisted_vectorstore = load_vectorstore()

    if selected_file is not None:
        new_db = faiss_indices[selected_file]
        vectors, vector_ids = extract_vectors_from_db(new_db)
    else:
        new_db = vectorStore

    embedding_start_time = time.time()
    query_vector = embedding.embed_query(user_question)
    embedding_end_time = time.time()
    embedding_time = (embedding_end_time - embedding_start_time) * 1000
    query_vector = np.array(query_vector).reshape(1, -1)

    query_vector = query_vector.reshape(1, -1)  # Ensure the shape is correct
    dimension = query_vector.shape[1]


    search_start_time = time.time()
    #docs,final_time = index_flat_search(user_question, new_db, vectors,vector_ids,query_vector,top_k=num_sources)
    #docs,final_time = ivf_index_search(user_question, new_db,vectors,vector_ids,query_vector)
    #docs,final_time = ivfpq_index_search(user_question, new_db,vectors,vector_ids,query_vector)
    #docs = hnsw_index_search(user_question, new_db,vectors,vector_ids)
    docs = new_db.similarity_search_with_relevance_scores(user_question)
    #docs,final_time = custom_search(user_question, new_db,vectors,vector_ids,query_vector)
    search_end_time = time.time()
    final_time = (search_end_time - search_start_time)* 1000

    #for doc in docs:
        #logging.info("-------------")
        #logging.info("Score: %s", doc[1])

    retriever = new_db.as_retriever()
    compressor = FlashrankRerank()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
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
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": st.session_state.conversation_memory,
        }
    )

    ai_message = qa.run(user_question)
    return ai_message,docs,final_time,embedding_time
   

def getDocNamesFromVectorStore(db):
        unique_docs = set()
        v_dict  = db.docstore._dict
        for k in v_dict.keys():
            doc_name = v_dict[k].metadata['source']
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


def show_vectorstore123(db):
    vector_dataframe = store_to_dataframe(db)
    st.write("Documents in Vector Store")
    st.write(vector_dataframe)

         

def show_vectorstore(db):
    vector_dataframe = store_to_dataframe(db)
    st.write(vector_dataframe)
    
    
def store_to_dataframe(db):
    v_dict  = db.docstore._dict
    data_rows=[]
    st.write("in delete.................")
    for key in v_dict.keys():
        doc_name = v_dict[key].metadata['source']
        page_number = v_dict[key].metadata['page']+1
        content = v_dict[key].page_content
        data_rows.append({"chunk_id":key, "document":doc_name, "page":page_number, "content":content})
        vector_df = pd.DataFrame(data_rows)
    return vector_df


def get_indices_for_file(db, file_name):
    documentList = []
    embedding = OpenAIEmbeddings(model="text-embedding-3-large")

    # Iterate through the existing vectorstore and filter embeddings for the specified file
    for key in db.docstore._dict.keys():
        doc_name = db.docstore._dict[key].metadata['source']
        
        if doc_name == file_name:
            # Append the embedding of the document to filtered_embeddings
            documentList.append(db.docstore._dict[key])
            #filtered_embeddings.append(db.docstore._dict[key].embedding)
    
    faiss_indices = {
                        str(file_name): FAISS.from_documents(documentList, embedding)
                    }
    # Create a new FAISS vectorstore from the filtered embeddings
    #new_vectorstore = FAISS.from_embeddings(filtered_embeddings)
    return faiss_indices


def index_flat_search(query, db_L2, vectors, vector_ids, query_vector, top_k=5):
    # Initialize Flat L2 index
    flat_index = faiss.IndexFlatL2(query_vector.shape[1])
    flat_index.add(np.array(vectors))  # Add vectors to the index

    search_start_time = time.time()
    # Search in the index
    D, I = flat_index.search(query_vector, top_k)

    alpha = 0.5
    # Combine relevancy score calculation and document retrieval in one loop
    flat_results = [(db_L2.docstore.search(vector_ids[idx]), np.exp(-alpha * dist)) 
                    for dist, idx in zip(D[0], I[0])]
    search_end_time = time.time()
    search_time = (search_end_time-search_start_time)

    return flat_results,search_time



def custom_search(query: str, db, vectors, vector_ids, query_vector, top_k: int = 5, cells_to_search: int = 10):
    # Ensure the query vector is correctly shaped

        search_start_time = time.time()

        # Step 3: Perform Flat L2 search within the collected vectors
        D, I = db.index.search(query_vector, top_k)  # Perform the search
        # Retrieve results from docstore using IDs
        alpha = 0.1
        # Retrieve results from docstore based on indices
        flat_results = [(db.docstore.search(vector_ids[idx]), np.exp(-alpha * dist)) 
                    for dist, idx in zip(D[0], I[0])]
        search_end_time = time.time()
        search_time = (search_end_time-search_start_time)

            #flat_results = [db_ivfpq.docstore.search(cell_ids[i]) for i in I[0]]
        return flat_results,search_time

        return []


def hybrid_search(query: str, db, vectors, vector_ids, query_vector, top_k: int = 5, cells_to_search: int = 10):
    # Ensure the query vector is correctly shaped
    query_vector = query_vector.reshape(1, -1)  # Ensure the shape is correct

    # Step 1: Search the db_ivfpq index for the closest cells
    distances, indices = db.index.search(query_vector, cells_to_search)
    

    # Step 2: Collect vectors and IDs from the top cells
    cell_vectors = []
    cell_ids = []
    for idx in indices[0]:
        if idx != -1:  # valid cell index
            cell_vectors.append(vectors[int(idx)])  # Retrieve from pre-stored vectors
            cell_ids.append(vector_ids[int(idx)])  # Retrieve from pre-stored vector IDs

    # Step 3: Perform Flat L2 search within the collected vectors
    if cell_vectors:
        dimension = query_vector.shape[1]
        flat_index = faiss.IndexFlatL2(dimension)
        flat_index.add(np.array(cell_vectors))  # Add collected vectors to the flat index
        D, I = flat_index.search(query_vector, top_k)  # Perform the search
        print(I)
        # Retrieve results from docstore using IDs
        alpha = 0.5
        # Retrieve results from docstore based on indices
        flat_results = [(db.docstore.search(cell_ids[idx]), np.exp(-alpha * dist)) 
                       for dist, idx in zip(D[0], I[0])]

        #flat_results = [db_ivfpq.docstore.search(cell_ids[i]) for i in I[0]]
        return flat_results

    return []

# Example usage:
# result = hybrid_search("sample query", db_ivfpq, vectors, vector_ids, query_vector)


def ivf_index_search(query: str, db_ivf, vectors, vector_ids, query_vector, nlist=50, top_k=5):
    dimension = query_vector.shape[1]
    # Create and train the IVF index with a Flat L2 quantizer
    quantizer = faiss.IndexFlatL2(dimension)  # Flat index for clustering
    ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)  # Using IVFFlat

    if vectors:
        ivf_index.train(np.array(vectors))
        ivf_index.add(np.array(vectors))

        # Perform search
        #ivf_index.nprobe = max(1, nlist // 10)  # Adjust nprobe for better search accuracy, often nlist/10 is a good start
        search_start_time = time.time()
  
        ivf_index.nprobe = 5
        D, I = ivf_index.search(query_vector, top_k)
        
        alpha = 0.5
        # Retrieve results from docstore based on indices
        ivf_results = [(db_ivf.docstore.search(vector_ids[idx]), np.exp(-alpha * dist)) 
                       for dist, idx in zip(D[0], I[0])]
        search_end_time = time.time()
        search_time = (search_end_time-search_start_time)
        return ivf_results,search_time

    return []
 

def ivfpq_index_search(query: str, db_ivfpq,vectors,vector_ids, query_vector,nlist=40, m=16, top_k=10):
    
    dimension = query_vector.shape[1]
    quantizer = faiss.IndexFlatL2(dimension)
    ivfpq_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 16, 8)  # 8 bits per sub-vector
    
    if vectors:
        ivfpq_index.train(np.array(vectors))
        ivfpq_index.add(np.array(vectors))


        search_start_time = time.time()
        #Perform IVFPQ search to get top results
        ivfpq_index.nprobe = 20  # Adjust nprobe for better search accuracy
        D, I = ivfpq_index.search(query_vector, top_k)

        alpha = 0.5
        ivf_results = [(db_ivfpq.docstore.search(vector_ids[idx]), np.exp(-alpha * dist)) 
                       for dist, idx in zip(D[0], I[0])]
        search_end_time = time.time()
        search_time = (search_end_time-search_start_time)

        return ivf_results,search_time

    return []



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



def extract_vectors_from_db(db):
    vectors = []
    ids = []
    for idx in range(len(db.index_to_docstore_id)):
        vectors.append(db.index.reconstruct(int(idx)))  # Reconstruct vector from index
        ids.append(db.index_to_docstore_id[int(idx)])  # Get corresponding docstore ID
    
    return vectors,ids
        

def main():
    db = None
    doc_list=[]
    if os.path.exists("db"):
        vector_start_time = time.time()
        db = load_vectorstore()
        vectors,vector_ids = extract_vectors_from_db(db)
        vector_end_time = time.time()
        vector_load_time = vector_end_time - vector_start_time
        print(f"Time taken to load vector: {vector_load_time} seconds")
        doc_list = getDocNamesFromVectorStore(db)

    #if db is not None:
     #show_vectorstore(db)
            

    if 'title' not in st.session_state:
        st.session_state.title = "RAG-Powered Custom ChatGPT"  # Default title

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
                st.write(chunkList)
            
                    
                # split the text chunks
                #text_chunks = get_text_chunks(pages)

                    # st.write("creatig vectorstore.....")
                db = load_vectorstore(chunkList)
                vectors,vector_ids = extract_vectors_from_db(db)

                # st.write(vectorStore)    

        files = ""
        # Add a slider for number of sources to return 1-5
        num_sources = st.slider("Number of sources per document:", min_value=1, max_value=5, value=3)
        index_type= ""
        #index_type = st.selectbox(
           # "Choose faiss index to search:",
           # options=["L2", "ivf", "ivfpq","hsnw","similarity_search"],
           # index=0  # Default to gpt-4o
       # )

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
                st.write(selected_file)

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
        query_start_time = time.time()
        model_response, context,final_time,embedding_time = model_query(user_question,num_sources,faiss_indices,selected_file,db,index_type,vectors,vector_ids)
        query_end_time = time.time()
        query_load_time = (query_end_time - query_start_time)
        print(f"Time taken to search: {final_time:} ms")

        print(f"Time taken for response: {query_load_time:} sec")

        print(f"Time taken for embedding: {embedding_time:.2f} ms")

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
                                    for doc,relevance in context:
                                        st.markdown(f"> **Context Document**: {doc.metadata['source']}")
                                        st.markdown(f"> **Page Number**: {doc.metadata['page']}")
                                        st.markdown(f"> **Content**: {doc.page_content}")
                                        st.markdown(f"> **Relevancy**: {relevance}")

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


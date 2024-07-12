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
logging.basicConfig(level=logging.INFO)
import time
import tiktoken

def load_vectorstore(text_chunks=None):
    if os.path.exists("db"):
        # Load the existing vectorstore
        db = FAISS.load_local("db", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        if text_chunks is not None:
            # Create a temporary vectorstore from the new text chunks
            new_vectorstore = FAISS.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings())
            # Merge the new vectorstore into the existing one
            db.merge_from(new_vectorstore)
            # Optionally save the updated vectorstore back to disk
            db.save_local("db")
    else:
        if text_chunks is not None:
            # Create a new vectorstore from the text chunks
            db = FAISS.from_documents(documents=text_chunks, embedding=OpenAIEmbeddings())
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
    #pattern = r'^(\d+\.\d* (?:\b\w+\b\s*){1,5}\w*)'
    #pattern = r'^\d+(\.\d+)?\s+[A-Z][a-zA-Z\s.]+$'
    #pattern = r"^\d+(?:\.\d+)* .*(?:\r?\n(?!\d+(?:\.\d+)* ).*)*"
    #pattern =  r"^\d+(\.)*(?:\.\d+)?\s+(.*)"
    #pattern = r"^\d+(?:\.\d+)* .*$"
    pattern =  r"^\d+(?:\.\d+)?\s+(.*)"
    #pattern =  r"^\d+(?:\.\d+)*\s+.*"
    #pattern = r"^\d+(?:\.\d+)*\s+[A-Za-z0-9\s]+(?:[^\n])*$"

    chunkList = []
    for file in files:
        page_count = 1
        source = file.name
        pdf_reader = PdfReader(file)  # it initialises pdf object which take pages
        text=0
        page_count = 0
        documents = []
        titles = ""

        for page in pdf_reader.pages:
            text = page.extract_text()  
            matches = re.findall(pattern, text, re.MULTILINE)

           
            if(len(matches)!=0):
                titles = matches

            page_content = text

            metadata = {
                'source' : source,
                'page': page_count,
                'heading': titles
            }
           

            pdf_data = {
                "page_content": page_content,
                "metadata": metadata
            }
           

            # Serialize the dictionary to a JSON string
            document = Document(json.dumps(pdf_data))
            page_count = page_count+1
            documents.append(document)

        text_chunks = get_text_chunks(documents)
        chunkList.extend(text_chunks)
    return chunkList    


def get_pdf_content_old(file):
    text = ""
    for docs in file:
        pdf_reader = PdfReader(docs)  # it initialises pdf object which take pages
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    return chunks 


def model_query(user_question,num_sources,faiss_indices,selected_file,vectorStore,index_type):
    # Gather related context from documents for each query
     # Load from local storage
    #persisted_vectorstore = load_vectorstore()

    if selected_file is not None:
        new_db = faiss_indices[selected_file]
    else:
        new_db = vectorStore

    extract_vectors_from_db(new_db)

    vectors,vector_ids  = extract_vectors_from_db(new_db)
    search_start_time = time.time()

    if index_type=="L2":
        docs = index_flat_search(user_question, new_db, vectors,vector_ids,top_k=num_sources)
        search_end_time = time.time()

    if index_type=="ivf":
        docs = ivf_index_search(user_question, new_db,vectors,vector_ids)
        search_end_time = time.time()

    if index_type=="ivfpq":
        docs = ivfpq_index_search(user_question, new_db,vectors,vector_ids)
        search_end_time = time.time()

    if index_type=="hsnw":
        docs = hnsw_index_search(user_question, new_db,vectors,vector_ids)
        search_end_time = time.time()

    if index_type=="similarity_search":
        docs = new_db.similarity_search_with_relevance_scores(user_question)
        search_end_time = time.time()


    final_time = (search_end_time - search_start_time)* 1000


    #for doc in docs:
        #logging.info("-------------")
        #logging.info("Score: %s", doc[1])

    retriever = new_db.as_retriever()

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
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
            "memory": st.session_state.conversation_memory,
        }
    )

    ai_message = qa.run(user_question)

    return ai_message,docs,final_time
   

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

def retrieve_and_display_embeddings(vectorstore):
    embeddings_list = []
    for key in vectorstore.index_to_docstore_id:
        doc_id = vectorstore.index_to_docstore_id[key]
        embedding = vectorstore.index.reconstruct(key)
        doc = vectorstore.docstore.search(doc_id)
        embeddings_list.append({
            "document_id": doc_id,
            "page": doc.metadata.get("page", ""),
            "source": doc.metadata.get("source", ""),
            "embedding": np.array(embedding)  # Ensure the embedding is a NumPy array
        })

    # Print or display the embeddings
    for item in embeddings_list:
        st.write(f"Document ID: {item['document_id']}")
        st.write(f"Page: {item['page']}")
        st.write(f"Source: {item['source']}")
        st.write("Embedding:")
        st.write(item['embedding'])
        st.write(f"Embedding Shape: {item['embedding'].shape}")  # Display the shape of the embedding
        st.write("\n")


def show_vectorstore123(db):
    vector_dataframe = store_to_dataframe(db)
    st.write("Documents in Vector Store")
    st.write(vector_dataframe)

    st.write("Embeddings in Vector Store")
    retrieve_and_display_embeddings(db)

         

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

def delete_document(faiss_index, db,document):
    st.write("-----------------")
    vector_dataframe = store_to_dataframe(faiss_index)
    chunks_list = vector_dataframe.loc[vector_dataframe['document']==document]['chunk_id'].tolist()
    st.write("&&&&&&&&&&&&&&&&&&&&&&")
    st.write(chunks_list)
    db.delete(chunks_list)


def get_indices_for_file(db, file_name):
    documentList = []
    # Iterate through the existing vectorstore and filter embeddings for the specified file
    for key in db.docstore._dict.keys():
        doc_name = db.docstore._dict[key].metadata['source']
        
        if doc_name == file_name:
            # Append the embedding of the document to filtered_embeddings
            documentList.append(db.docstore._dict[key])
            #filtered_embeddings.append(db.docstore._dict[key].embedding)
    
    faiss_indices = {
                        str(file_name): FAISS.from_documents(documentList, OpenAIEmbeddings(disallowed_special=()))
                    }
    # Create a new FAISS vectorstore from the filtered embeddings
    #new_vectorstore = FAISS.from_embeddings(filtered_embeddings)
    return faiss_indices


def index_flat_search(query, db_L2,vectors,vector_ids, top_k=5):
    embedding = OpenAIEmbeddings()
    query_vector = embedding.embed_query(query)
    query_vector = np.array(query_vector).reshape(1, -1)

    # Perform Flat L2 search within the collected vectors
    flat_index = faiss.IndexFlatL2(query_vector.shape[1])
    vectors = [db_L2.index.reconstruct(i) for i in range(db_L2.index.ntotal)]
    flat_index.add(np.array(vectors))

    D, I = flat_index.search(query_vector, top_k)

    alpha = 0.5
    relevancy_scores = [np.exp(-alpha * d) for d in D[0]]
        
    flat_results = [(db_L2.docstore.search(vector_ids[i]), relevancy_scores[j]) for j, i in enumerate(I[0])]

    return flat_results

def create_ivf_index(vectors, d, nlist=100):
    quantizer = faiss.IndexFlatL2(d)  # Flat index for clustering
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    
    # Train the IVF index
    index.train(vectors)
    index.add(vectors)

    return index

def hybrid_search(query: str, db_ivfpq, top_k: int = 5, cells_to_search: int = 6):
    embedding = OpenAIEmbeddings()
    
    # Step 1: Perform IVFPQ search to get top cells
    query_vector = embedding.embed_query(query)
    query_vector = np.array(query_vector)  # Convert to NumPy array
    query_vector = query_vector.reshape(1, -1)  # Ensure the shape is correct

    distances, indices = db_ivfpq.index.search(query_vector, cells_to_search)

    # Step 2: Collect vectors from the top cells
    cell_vectors = []
    cell_ids = []
    for idx in indices[0]:
        if idx != -1:  # valid cell index
            cell_vectors.append(db_ivfpq.index.reconstruct(int(idx)))  # Ensure idx is an integer
            cell_ids.append(db_ivfpq.index_to_docstore_id[int(idx)])  # Ensure idx is an integer

    # Step 3: Perform Flat L2 search within the collected vectors
    if cell_vectors:
        dimension = query_vector.shape[1]
        flat_index = faiss.IndexFlatL2(dimension)
        flat_index.add(np.array(cell_vectors))
        D, I = flat_index.search(query_vector, top_k)
        flat_results = [db_ivfpq.docstore.search(cell_ids[i]) for i in I[0]]
        return flat_results

    return []

def ivf_index_search(query: str, db_ivf, vectors,vector_ids, nlist=100, top_k=5):
    embedding = OpenAIEmbeddings()
    
    # Step 1: Perform IVFPQ search to get top cells
    query_vector = embedding.embed_query(query)
    query_vector = np.array(query_vector).reshape(1, -1)  # Ensure the shape is correct

    dimension = query_vector.shape[1]
    # Create and train the IVF index
    quantizer = faiss.IndexFlatL2(dimension)  # Flat index for clustering
    ivf_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, 8, 8)  # Using IVFPQ with 8 bits per sub-vector


    if vectors:
        ivf_index.train(np.array(vectors))
        ivf_index.add(np.array(vectors))
        
        D, I = ivf_index.search(query_vector, top_k)
        # Calculate relevancy scores from distances
        alpha = 0.5
        relevancy_scores = [np.exp(-alpha * d) for d in D[0]]
        
        # Retrieve results from docstore based on indices
        ivf_results = [(db_ivf.docstore.search(vector_ids[i]), relevancy_scores[j]) for j, i in enumerate(I[0])]
        return ivf_results

    return []   

def hnsw_index_search(query: str, db_ivf, vectors, vector_ids, top_k=5, ef_search=50, ef_construction=200):
    embedding = OpenAIEmbeddings()
    
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


def hsnw_index_search(query: str, db_ivf, vectors,vector_ids, nlist=100, top_k=5):
    d = 128      # Dimension (length) of vectors.
    M = 32       # Number of connections that would be made for each new vertex during HNSW construction.

    # Creating the index.
    index = faiss.IndexHNSWFlat(d, M)            
    index.hnsw.efConstruction = 40         # Setting the value for efConstruction.
    index.hnsw.efSearch = 16               # Setting the value for efSearch.

    # Adding vectors to the index (xb are database vectors that are to be indexed).
    index.add(vectors)                  

    # xq are query vectors, for which we need to search in xb to find the k nearest neighbors.
    # The search returns D, the pairwise distances, and I, the indices of the nearest neighbors.
    D, I = index.search(xq, k)    

def extract_vectors_from_db(db):
    vectors = []
    ids = []
    for idx in range(len(db.index_to_docstore_id)):
        vectors.append(db.index.reconstruct(int(idx)))  # Reconstruct vector from index
        ids.append(db.index_to_docstore_id[int(idx)])  # Get corresponding docstore ID
    
    return vectors,ids
        

def ivfpq_index_search(query: str, db_ivfpq,vectors,vector_ids, nlist=100, m=8, top_k=5):
    
    embedding = OpenAIEmbeddings()
    query_vector = embedding.embed_query(query)
    query_vector = np.array(query_vector).reshape(1, -1)

    dimension = query_vector.shape[1]
    quantizer = faiss.IndexFlatL2(dimension)
    ivfpq_index = faiss.IndexIVFPQ(quantizer, dimension, nlist, m, 8)  # 8 bits per sub-vector
    
    if vectors:
        ivfpq_index.train(np.array(vectors))
        ivfpq_index.add(np.array(vectors))

        #Perform IVFPQ search to get top results
        ivfpq_index.nprobe = 10  # Adjust nprobe for better search accuracy
        D, I = ivfpq_index.search(query_vector, top_k)

        alpha = 0.5
        relevancy_scores = [np.exp(-alpha * d) for d in D[0]]

        # Retrieve results from docstore based on indices
        ivfpq_results = [(db_ivfpq.docstore.search(vector_ids[i]), relevancy_scores[j]) for j, i in enumerate(I[0])]
        return ivfpq_results

    return []


def main():
    load_dotenv()
    db = None
    doc_list=[]
    if os.path.exists("db"):
        vector_start_time = time.time()
        db = load_vectorstore()
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

    

    st.set_page_config(st.session_state.title,page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    vectorStore = None
    faiss_indices  = None
    with st.sidebar:
        st.title("Document Query Settings")        
        uploaded_file_names = []
        files = st.file_uploader("Upload your PDFs here and click on process", accept_multiple_files=True)
        text_chunks = ""
        append_chunks = ""
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
                st.write(selected_file)

            if st.button("delete document"):
                delete_document(faiss_indices[selected_file],db,selected_file)
                
            if st.button("Summary"):
                st.session_state.is_summary = True
                with st.spinner("Processing..."):

                    if selected_file is None:
                        st.error(f"Error: Please select the file to create summary.")
                    
                    
                    template_string = """
                    Read the text delimited by three angular brackets. These are the chunks the documents have been split into.
                    Step 1: Summarize Each Chunk
                    Instruction: Please summarize each chunk individually, capturing the main themes and key points. Do not make things up; only summarize what is actually present in the text. No hallucination.
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
                        text = ""
                        newdb = faiss_indices[selected_file]

                        for key in newdb.docstore._dict.keys():
                            doc = newdb.docstore._dict[key]
                            all_chunks.append(doc)

                        docs = faiss_indices[selected_file].similarity_search_with_relevance_scores(question)
                      
                        #st.write(docs)  
                        #st.write(len(docs))
                       # Summarize chunks in parallel
                        start_time = time.time()
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            chunk_summaries = list(executor.map(summarize_chunk, all_chunks))

                        # Filter out None values (in case of errors)
                        chunk_summaries = [summary for summary in chunk_summaries if summary]

                        # Final template for creating a summary from chunk summaries
                        final_template_string = """
                        Based on the summaries of all the chunks provided below, create a final, concise, and to-the-point summary that captures the main themes and key points from the entire document. Exclude the disclaimer from the summary if any exists in the document.
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
    #with st.expander("Show VectorStore"):
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
        model_response, context,final_time = model_query(user_question,num_sources,faiss_indices,selected_file,db,index_type)
        query_end_time = time.time()
        query_load_time = query_end_time - query_start_time
        print(f"Time taken to search: {final_time:.2f} ms")

        print(f"Time taken for response: {query_load_time} seconds")


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


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
import os
import re
import json
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

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
            st.write("chubk isnot noe........")
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
 

def get_pdf_content(file):
    #pattern = r'^(\d+\.\d* (?:\b\w+\b\s*){1,5}\w*)'
    #pattern = r'^\d+(\.\d+)?\s+[A-Z][a-zA-Z\s.]+$'
    #pattern = r"^\d+(?:\.\d+)* .*(?:\r?\n(?!\d+(?:\.\d+)* ).*)*"
    #pattern =  r"^\d+(\.)*(?:\.\d+)?\s+(.*)"
    #pattern = r"^\d+(?:\.\d+)* .*$"
    pattern =  r"^\d+(?:\.\d+)?\s+(.*)"
    #pattern =  r"^\d+(?:\.\d+)*\s+.*"
    #pattern = r"^\d+(?:\.\d+)*\s+[A-Za-z0-9\s]+(?:[^\n])*$"

    page_count = 1
    for docs in file:
        source = docs.name
        pdf_reader = PdfReader(docs)  # it initialises pdf object which take pages
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
    return documents    


def get_pdf_content_old(file):
    text = ""
    for docs in file:
        pdf_reader = PdfReader(docs)  # it initialises pdf object which take pages
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(pages):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["(?<=\.)"," "],
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_documents(pages)
    return chunks 



def get_conversational_chain(vectorStore,input_llm):

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=input_llm, 
        retriever = vectorStore.as_retriever(),
        memory = memory
        )
    return conversation_chain



def model_query(user_question,num_sources,faiss_indices,selected_file):
    # Gather related context from documents for each query
     # Load from local storage
    #persisted_vectorstore = load_vectorstore()
    new_db = faiss_indices[selected_file]
    docs = new_db.similarity_search_with_relevance_scores(user_question, k=num_sources)
    for doc in docs:
        st.write(doc)
        logging.info("-------------")
        logging.info("Score: %s", doc[1])

    retriever = new_db.as_retriever()

    #rileysPrompt()
    #prompt fot 
    template = """
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


    return ai_message,docs
   

def getDocNamesFromVectorStore(db):
        unique_docs = set()
        v_dict  = db.docstore._dict
        for k in v_dict.keys():
            doc_name = v_dict[k].metadata['source']
            unique_docs.add(doc_name)
        unique_docs_list = list(unique_docs)
        return unique_docs_list

def isFilesExist(uploaded_file_name,unique_docs_list):
        
        if uploaded_file_name in unique_docs_list:
            st.error(f"Error: {uploaded_file_name} already exists in the vector store.")
            return False
        else:
           unique_docs_list.append(uploaded_file_name)
           return True
           
               
            
         
def show_vectorstore(db):
    vector_dataframe = store_to_dataframe(db)
    st.write(vector_dataframe)
    
def store_to_dataframe(db):
    v_dict  = db.docstore._dict
    data_rows=[]
    for key in v_dict.keys():
        doc_name = v_dict[key].metadata['source']
        page_number = v_dict[key].metadata['page']+1
        content = v_dict[key].page_content
        data_rows.append({"chunk_id":key, "document":doc_name, "page":page_number, "content":content})
        vector_df = pd.DataFrame(data_rows)
    return vector_df



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




def main():
    load_dotenv()

    doc_list=[]
    if os.path.exists("db"):
        db = load_vectorstore()
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

        st.subheader("Your Documents")
        
        uploaded_file_names = ""
        files = st.file_uploader("Upload your PDFs here and click on process",accept_multiple_files=True)

        if st.button("Load Data"):
            if files:
                st.session_state.uploaded_file=True
                for f in files:
                    uploaded_file_names = f.name

            if doc_list is not None:
                isFile = isFilesExist(uploaded_file_names,doc_list)
            
            if isFile:
                for f in files:
                    # get content from uploaded PDFs
                    pages = get_pdf_content(files)
                    
                    
                    # split the text chunks 
                    text_chunks = get_text_chunks(pages)
                    

                    #st.write("creatig vectorstore.....")
                    vectorStore = load_vectorstore(text_chunks) 
                    #st.write(vectorStore)
        #selected_files = st.multiselect("Please select the files to query:", options=doc_list)
        selected_file = st.selectbox(
                    "Please select the files to query:",
                    options=doc_list,
                    index=0  # Default to gpt-4o
                )
        
        # Add a slider for number of sources to return 1-5
        num_sources = st.slider("Number of sources per document:", min_value=1, max_value=5, value=3)

        #temp_value = st.text_input("Mention a Temperature range(0 to 1)(Default:0)")
        
        model_version = st.selectbox(
            "Select the GPT model version:",
            options=["gpt-4o","gpt-4-1106-preview","gpt-3.5-turbo-1106"],
            index=0  # Default to gpt-4o
        )

        st.session_state.model = model_version
        llm = ChatOpenAI(temperature =0,model = st.session_state.model)
        if selected_file:
         vectorStore = load_vectorstore() 
         faiss_indices = get_indices_for_file(vectorStore,selected_file)
                
        if st.button("Summary"):
            st.session_state.is_summary = True
            with st.spinner("Processing..."):
                st.write("........................")

                if selected_file is None:
                    st.error(f"Error: Please select the file to create summary.")
                
                template_string = "read the text delimited by three angular brackets.\
                     your task is to read the demited list of text, extract most relevent information which summarize the whole document.\
                    ,starting with 'the uploaded document is..'\
                     and than ask user what else they want to know about that document?\
                    <<<{text}>>>  "
                     #Are you sure that''s your final answer?\
                     #It might be worth taking another look\
                    #Remember that progress is made one step at a time.\
                     #Stay determined and keep moving forward.\
            

                #show_vectorstore(new_vectorstore)
            
                question = "give me summary of this uploaded document."
                #if len(pages) >= 10:
                    #k = 10
                #else:
                    #k = 5
                #st.write(faiss_indices[selected_file])
                if faiss_indices is not None:

                    docs = faiss_indices[selected_file].similarity_search_with_relevance_scores(question,k=10)

                    prompt_template = ChatPromptTemplate.from_template(template_string)

                    customer_messages = prompt_template.format_messages(
                    text=docs
                    )

                    summary_response = llm(customer_messages)
                    st.session_state.summary = summary_response
                    #st.write(summary_response.content)

            
    # Main section
    st.title(st.session_state.title)


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
        model_response, context = model_query(user_question,num_sources,faiss_indices,selected_file)
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
                    st.write(st.session_state.summary.content)
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
                                        st.markdown(f"> **relevancy**: {doc[1]}")

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
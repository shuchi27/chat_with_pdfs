import streamlit as st
from dotenv import load_dotenv 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import  css, bot_template, user_template
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain.vectorstores import Chroma
import os
import re
import json

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
    text = ""
    for docs in files:
        pdf_reader = PdfReader(docs)  # it initialises pdf object which take pages
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text    

def get_pdf_content2(file):
    #pattern = r'^(\d+\.\d* (?:\b\w+\b\s*){1,5}\w*)'
    #pattern = r'^\d+(\.\d+)?\s+[A-Z][a-zA-Z\s.]+$'
    #pattern = r"^\d+(?:\.\d+)* .*(?:\r?\n(?!\d+(?:\.\d+)* ).*)*"
    pattern = r"^\d+(?:\.\d+)* .*$"
   #pattern =  r"^\d+(?:\.\d+)?\s+(.*)"

    page_count = 1
    for docs in file:
        pdf_reader = PdfReader(docs)  # it initialises pdf object which take pages
        #st.write(pdf_reader)
        text=0
        page_count = 0
        documents = []
        for page in pdf_reader.pages:
            text = page.extract_text()
            
            matches = re.findall(pattern, text, re.MULTILINE)
            
            if(len(matches)!=0):
                titles = matches
            st.write(titles)
            page_content = text
            metadata = {
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
        #st.write(documents)
    return documents    

def get_pdf_content_old(files):
    text = ""
    for docs in files:
        pdf_reader = PdfReader(docs)  # it initialises pdf object which take pages
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(pages):
    text_splitter = CharacterTextSplitter(
        separator=" ",
        chunk_size=1000,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(pages)
    return chunks 


def get_vectorstore(text_chunks):
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    persist_directory = 'docs/chroma/'
    embeddings = OpenAIEmbeddings()
    #rm -rf ./docs/chroma  # remove old database files if any
    #vectorstore = Chroma.from_documents(documents=text_chunks,embedding=embeddings,persist_directory=persist_directory)
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversational_chain(vectorStore,input_llm):

    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=input_llm, 
        retriever = vectorStore.as_retriever(),
        memory = memory
        )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

   # Reverse the chat history list
    chat_history_reversed = st.session_state.chat_history[::-1]

    for i, message in reversed(list(enumerate(st.session_state.chat_history))):

        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


    #st.write(response)

def main():
    load_dotenv()
    st.set_page_config(os.environ['page_title'],page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    temp_value = os.environ['temperature']
    model_value = os.environ['model']
    llm = ChatOpenAI(temperature=temp_value,model=model_value)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    st.header(os.environ['page_title'])
    footer = st.empty()

    user_question = footer.text_input("Ask a question")
    if user_question:
       handle_userinput(user_question)
    
    #st.write(user_template.replace("{{MSG}}", "Hello Robot"),unsafe_allow_html=True)
    #st.write(bot_template.replace("{{MSG}}", "Hello Human"),unsafe_allow_html=True)
    #user_template = st.empty()
    #bot_template = st.empty()
    with st.sidebar:
        st.subheader("Your Documents")
        files = st.file_uploader("Upload your PDFs here and click on process",accept_multiple_files=True)
        
        model_option = st.selectbox('Select Model:',('gpt-4', 'gpt-3.5-turbo'))
        
        temp_value = st.text_input("Mention a Temperature range(0 to 1)(Default:0.9)")
        
        if st.button("Process"):
            with st.spinner("Processing..."):
              #st.session_state.conversation = None
             # user_template.empty()
             # bot_template.empty()

              if not files:
                 st.error("Please upload documents.")

              else:
                if model_option is not None:
                    os.environ['model'] = model_option

                #st.write("len:",len(temp_value))
                if len(temp_value) != 0:
                    os.environ['temperature'] = temp_value
                else:
                    os.environ['temperature'] = "0.0"


                # get content from uploaded PDFs
                pages = get_pdf_content(files)
                

                template_string = "read the text delimited by three angular brackets.\
                     your task is to read the demited list of text, extract most relevent information which summarize the whole document.\
                     write that into less than 50 words, starting with 'the uploaded document is..'\
                     and than ask user what else they want to know about that document?\
                    <<<{text}>>>  "

                     #Are you sure that''s your final answer?\
                     #It might be worth taking another look\
                    #Remember that progress is made one step at a time.\
                     #Stay determined and keep moving forward.\

                # split the text chunks 
                text_chunks = get_text_chunks(pages)
                pattern = r"^\d+(?:\.\d+)* .*$"

                for chunk in text_chunks:
                    matches = re.findall(pattern, chunk, re.MULTILINE)
                    if(len(matches)!=0):
                         titles = matches
                         st.write(chunk)
                         st.write(titles)
                
                
                
                st.write(text_chunks)

                # create vector store 
                vectorStore = get_vectorstore(text_chunks)
                #st.write(vectorStore._collection.count())
                question = "give me summary of this uploaded document?for that look for conclusion and introduction part of the document if those exist in the document"
                st.write("******************************************")
                if len(pages) >= 10:
                    k = 10
                else:
                    k = 3
                docs = vectorStore.similarity_search(question,k=k)
                st.write(docs)


                prompt_template = ChatPromptTemplate.from_template(template_string)
               # st.write(prompt_template.messages[0].prompt)
                custom_msg = prompt_template.format_messages(text=docs)
                summary_response =llm(custom_msg)
                st.write(summary_response.content)
                st.write(prompt_template.messages[0].prompt.input_variables)

                #converstational chain
                #st.session_state.conversation = get_conversational_chain(vectorStore,llm)



if __name__ == '__main__':
    main()
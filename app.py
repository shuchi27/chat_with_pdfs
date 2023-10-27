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
from langchain.prompts import PromptTemplate
import os

def get_pdf_content(uploaded_docs):
    text = ""
    for docs in uploaded_docs:
        pdf_reader = PdfReader(docs)  # it initialises pdf object which take pages
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(pdf_text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators="",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
      )
    chunks = text_splitter.split_text(pdf_text)
    return chunks  

def get_vectorstore(text_chunks):
    #embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversational_chain(vectorStore):
    temp_value = os.environ['temperature']
    model_value = os.environ['model']

    #llm = ChatOpenAI(temperature=0.9,model="gpt-4")
    st.write("Temperature==",temp_value)
    st.write("Model=", model_value)
    

    llm = ChatOpenAI(temperature=temp_value,model=model_value)
    memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
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
        uploaded_docs = st.file_uploader("Upload your PDFs here and click on process",accept_multiple_files=True)
        
        model_option = st.selectbox('Select Model:',('gpt-4', 'gpt-3.5-turbo'))
        
        temp_value = st.text_input("Mention a Temperature range(0 to 1)(Default:0.9)")
        
        if st.button("Process"):
            with st.spinner("Processing..."):
              #st.session_state.conversation = None
             # user_template.empty()
             # bot_template.empty()

              if not uploaded_docs:
                 st.error("Please upload documents.")

              else:
                if model_option is not None:
                    os.environ['model'] = model_option

                #st.write("len:",len(temp_value))
                if len(temp_value) != 0:
                    os.environ['temperature'] = temp_value
                else:
                    os.environ['temperature'] = "0.9"


                # get content from uploaded PDFs
                pdf_text = get_pdf_content(uploaded_docs)
                #st.write(pdf_text)


                # split the text chunks 
                text_chunks = get_text_chunks(pdf_text)
                #st.write(text_chunks)

                # create vector store 
                vectorStore = get_vectorstore(text_chunks)

                #converstational chain
                st.session_state.conversation = get_conversational_chain(vectorStore)



if __name__ == '__main__':
    main()
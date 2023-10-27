# from dotenv import load_dotenv
import os
import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader

def main():

    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv()) # read local .env file

    openai.api_key  = os.environ['OPENAI_API_KEY']
    # load_dotenv()
   # os.environ["OPENAI_API_KEY"] = "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"
    llm = OpenAI(temperature=0.1)

    st.set_page_config(os.environ['page_title'])
    st.header(os.environ['header'])
    
    # upload file
    pdf = st.file_uploader("Upload your Document", type="pdf" )

    #loader = PyPDFLoader("docs/cs229_lectures/MachineLearning-Lecture01.pdf")
    #pages = loader.load() 
    #each page is a document and A `Document` contains text (`page_content`) and `metadata`.

    # extract the text
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
      
      
      # split into chunks
      
      text_splitter = RecursiveCharacterTextSplitter(
        separators="",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
      )
      splits = text_splitter.split_text(text)
      
      st.write(splits)
      
      # create embeddings
      embeddings = OpenAIEmbeddings()

      #client = chromadb.PersistentClient(path="./db")

      persist_directory = '/Users/Desktop/SDSU/docs/chroma/'


      #vectordb = Chroma.from_documents(splits, embeddings, metadatas=[{"source": f"Text chunk {i} of {len(splits)}"} for i in range(len(splits))], persist_directory="db")
     # vectordb = Chroma.from_documents(splits, embeddings)

      #vectordb.persist()
      vectordb = FAISS.from_texts(splits, embeddings)
       # show user input
      user_question = st.text_input("How can I help you?")
      st.write(user_question)
      if user_question:
        document = vectordb.similarity_search(user_question,k=3)
      
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=document, question=user_question)
          print(cb)
           
          st.write(response)
    

if __name__ == '__main__':
    main()

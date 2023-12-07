import streamlit as st
from io import StringIO
import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

load_dotenv()

uploaded_file = st.file_uploader("Choose a py file", "py")
question = st.text_input("Enter your query:")
openai_api_key = os.environ.get("OPENAI_API_KEY")
if st.button('Submit'):
    if uploaded_file is not None and question != "":
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=20,
            length_function=len
        )

        texts = text_splitter.split_text(string_data)

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        vector_store = DocArrayInMemorySearch.from_texts(texts, embeddings)

        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(), chain_type="stuff", retriever=vector_store.as_retriever(), return_source_documents=True
        )

        result = qa({"query": question})

        st.write(result["result"])
    else:
        st.error("Please upload a document and enter a query!")
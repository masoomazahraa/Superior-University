from flask import Flask, request, render_template
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

app=Flask(__name__)

loader=DirectoryLoader('data',glob='*.pdf',loader_cls=PyPDFLoader)
documents=loader.load()
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text=text_splitter.split_documents(documents)
embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectordb=Chroma.from_documents(text,embeddings,persist_directory='./chorma_db')
retriever=vectordb.as_retriever()
llm= ChatGroq(
    temperature=0,
    groq_api_key = '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',
    model_name='llama3-70b-8192'
)
prompt_template="""You are a Compassionate mental health support chatbot. respond throughtfully to the following question:
    {context}
    user:{question}
    Chatbot"""
prompt=PromptTemplate(template=prompt_template,input_variables=['context','question'])
qa_chain=RetrievalQA.from_chain_type(llm=llm, chain_type='stuff',retriever=retriever, chain_type_kwargs={'prompt':prompt})

@app.route('/', methods=['GET','POST'])
def index():
    answer=None
    if request.method=='POST':
        question=request.form.get('question')
        if question:
            answer=qa_chain.run(question)
    return render_template('index.html',answer=answer)

if __name__=='__main__':
    app.run(debug=True)


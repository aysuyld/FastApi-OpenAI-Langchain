from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
from typing import List
import getpass
from pdfminer.high_level import extract_text_to_fp
from io import BytesIO
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import uvicorn
import os

os.environ['OPENAI_API_KEY'] = getpass.getpass('OpenAI API Key:')

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload", status_code=201)
async def upload_file(file: UploadFile):
    global vectore_store
    global texts
    try:
        content = await file.read()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        embeddings = OpenAIEmbeddings()
        texts = text_splitter.split_text(content)
        vectore_store = Chroma.from_documents(texts, embeddings)

        return {"message": "Upload successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during file upload")
    
@app.post("/query", status_code=200)
async def query_documents(question: str):
    try:
        #chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
        #chain.run(input_documents=texts, query=question)
        docs = vectore_store.similarity_search(question)
            
        return {"answer": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during query")


if __name__ == "__main__":
    uvicorn.run("main:app", host ="127.0.0.1", reload=True)
import os
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import SKLearnVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
import re
import torch
from transformers import AutoModelForCausalLM
import numpy as np
import streamlit as st

def load_llama_model():
    device = 'cuda'
    model_id = "/home/hice1/dbabu6/scratch/Llama-3.1-8B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256, temperature = 0.2)

LLM = load_llama_model()

def text_split():
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0)
    return text_splitter

text_splitter = text_split()

file_path = "/home/hice1/dbabu6/scratch/pdf_files/PDF_Syllabus_Dataset"

def load_documents():
    docs = []
    for file in os.listdir(file_path):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(file_path, file))
                pdf_docs = loader.load()
                docs.extend(pdf_docs)
                # logger.info(f"Loaded document: {file}")
            except Exception as e:
                continue
                # logger.error(f"Error loading {file}: {e}")
    # logger.info(f"Total documents loaded: {len(docs)}")
    return docs

docs = load_documents()

doc_splits = text_splitter.split_documents(docs)


### setting up the prompt template ###
prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks based on course content of Georgia Tech ECE department.
    Use the following documents to answer the question.
    Use five sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

#### initialize the embedding model #### 

def setup_embedding():
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings' : True}
    model_norm = HuggingFaceBgeEmbeddings(model_name= model_name,
    model_kwargs = {'device' : 'cuda'}, encode_kwargs = encode_kwargs)

    #### initializing the vectorstore ####
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=model_norm,
    )
    retriever = vectorstore.as_retriever(k=3)
    return retriever

retriever = setup_embedding()

def setup_ragchain():
# Create an LLM wrapper for your Hugging Face pipeline
    llm = HuggingFacePipeline(pipeline=LLM)

    # Create the LLMChain
    rag_chain = LLMChain(llm=llm, prompt=prompt)

    return rag_chain

rag_chain = setup_ragchain()

class RAGapplication():
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        docs = self.retriever.invoke(question)
        docs_texts = "\n".join([str(n.page_content) for n in docs])
        # print(docs_texts)
        input_data = {
            "question": str(question),
            "documents": docs_texts
        }
        answer = self.rag_chain.invoke(input_data)
        return answer
    
rag_app = RAGapplication(retriever = retriever, rag_chain = rag_chain)

def get_answer(question):
    question = str(question)
    answer = rag_app.run(question)
    answer = answer['text']
    return answer

def extract_answer(text):
    answer_start = text.find("Answer:")
    if answer_start == -1:
        return "Answer not found."
    answer = text[answer_start + len("Answer:"):].strip()
    return answer
# ans_processed = extract_answer(ans)

st.title('Welcome to Gatech ECE course query')
st.image('/home/hice1/dbabu6/scratch/gt_ece.png')

question = st.text_input('Ask a question:')

if question:
    answer = get_answer(question)
    answer_processed = extract_answer(answer)
    st.write("Response:", answer_processed)


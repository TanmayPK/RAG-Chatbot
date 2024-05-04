import os
import gc
import streamlit as st
import tempfile
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SummaryIndex
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.core.llms import ChatMessage

from llama_index.core.llms import ChatMessage

rag_flag: bool = False

def clear_conversation():
    st.session_state.messages = []
    st.session_state.query_engine.reset()
    if st.session_state.query_engine_file:
        st.session_state.query_engine_file.reset()
    gc.collect()



def initialize_query_engine():
    llm=Ollama(model="llama3", request_timeout=120.0)
    st.session_state.llm = llm
    index = SummaryIndex([])#VectorStoreIndex([])
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)
    Settings.embed_model = embed_model
    Settings.llm = llm
    
    st.session_state.query_engine = index.as_chat_engine(chat_mode= 'react', streaming=True)


with st.sidebar:
    
    st.header(f"Upload PDF files here")
    
    temp_file = st.file_uploader("Choose a `.pdf` file", type="pdf")
    
    if temp_file:
        rag_flag = True
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, temp_file.name)
            with open(file_path, "wb") as f:
                f.write(temp_file.read())
                
            st.write(f"Uploaded file: {temp_file.name}")
            
            if "query_engine_file" not in st.session_state or st.session_state.query_engine_file is None:             
                documents = SimpleDirectoryReader(temp_dir, required_exts=[".pdf"], recursive=True).load_data()
                vector_index = VectorStoreIndex.from_documents(documents, show_progress=True)
                st.session_state.query_engine_file = vector_index.as_chat_engine(chat_mode='react', streaming = True)
                    
                
                    
                
                st.success("Indexing complete!")
                rag_flag = True
    else:
        st.session_state.query_engine_file = None
        rag_flag = False
            
            
            
col1, col2 = st.columns([6, 2])

with col1:
    st.header(f"RAG-LLM Chatbot - Llama 3")

with col2:
    st.button("Clear conversation", on_click=clear_conversation)

if st.session_state.get("query_engine", None) is None:
    initialize_query_engine()

if "messages" not in st.session_state:
    rag_flag = False
    st.session_state.file_cache = {}
    clear_conversation()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
if prompt := st.chat_input("Enter your query here"):
    
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    messages = [ChatMessage(role=message['role'], content=message['content']) for message in st.session_state.get("messages", [])]
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        
        empty_message = st.empty()
        resp = ""
        if rag_flag:
            streamed_resp = st.session_state.query_engine_file.stream_chat(prompt)
        else:
            #prompt += "\n Do not use a tool to answer."
            #streamed_resp = st.session_state.query_engine.stream_chat(prompt)
            streamed_resp = st.session_state.llm.stream_chat(messages)
            
            
        if rag_flag:
            for chunk in streamed_resp.response_gen:
                resp += chunk
                empty_message.markdown(resp + "...")
        else:
            
            for chunk in streamed_resp:
                resp += chunk.delta
                empty_message.markdown(resp + "...")
            
        empty_message.markdown(resp)
    
    st.session_state.messages.append({"role": "assistant", "content": resp})
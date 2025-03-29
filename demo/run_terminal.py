import os
import re
import sys
import json
import dill  
import jieba
import pickle

from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings

from FlagEmbedding import LayerWiseFlagLLMReranker
from IPython.display import Image, display

current_dir = os.path.dirname(__file__)  
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src import config
from src.dag.workflow import build_workflow

DRAW_PIPLINE=False

def custom_preprocess_func(text):
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]+", " ", text)
        tokens = jieba.lcut(text)
        return [tok.strip() for tok in tokens if tok.strip()]

def init_rag():
    local_llm_name = 'qwen2.5:14b'
    llm = ChatOllama(model=local_llm_name, temperature=0.7)
    llm_json_mode = ChatOllama(model=local_llm_name, temperature=0.7, format='json')
    
    embedding_model_name = "bge-m3:latest"
    embedding = OllamaEmbeddings(model=embedding_model_name)

    sparse_index_path = "./vectordb/bm25_index.pkl"
    dense_faiss_dir = "./vectordb/faiss_index" 

    vectorstore = FAISS.load_local(dense_faiss_dir, embedding, allow_dangerous_deserialization=True)
    dense_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    sparse_retriever = pickle.load(open(sparse_index_path, "rb"))
    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever], weights=[0.5, 0.5]
    )

    reranker_model_name = "/mnt/workspace/pretrain_model/BAAI/bge-reranker-v2-minicpm-layerwise/"
    reranker = LayerWiseFlagLLMReranker(reranker_model_name, use_fp16=True)

    web_search_tool = TavilySearchResults(k=3)

    return llm, llm_json_mode, ensemble_retriever, reranker, web_search_tool


def main():
    llm, llm_json_mode, ensemble_retriever, reranker, web_search_tool = init_rag()
    graph = build_workflow(llm, llm_json_mode, ensemble_retriever, reranker, web_search_tool)

    if DRAW_PIPLINE:
        graph_image = graph.get_graph().draw_mermaid_png()        
        with open("workflow.png", "wb") as f:
            f.write(graph_image)
        print(f"图片已保存至 {image_path}")
        
    config = {"configurable": {"thread_id": "2"}}

    input_message = [
        SystemMessage(content="You are Jarvis, created by Baoxin wang. You are a helpful assistant."), 
        HumanMessage(content="今天北京的天气如何？")]
    
    inputs = {"messages": input_message, "max_retries": 3, "log":"ROUTE QUESTION"}
    
    for event in graph.stream(inputs, config, stream_mode="values"):
        # pass
        print(event)

if __name__ == "__main__":
    main()
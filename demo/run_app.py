import os
import re
import sys
import json
import dill
import time
import torch
import jieba
import pickle

import streamlit as st
from datetime import datetime
from typing import List, Dict, Tuple, Optional

from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.schema import SystemMessage, HumanMessage, AIMessage, FunctionMessage

from FlagEmbedding import LayerWiseFlagLLMReranker

current_dir = os.path.dirname(__file__)  
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)

from src import config
from src.dag.workflow import build_workflow

# 手动修正 torch.classes.__path__，以解决某些 Python/Faiss 的兼容性问题
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# ========== (1) Streamlit 的全局资源初始化 ==========

def custom_preprocess_func(text):
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]+", " ", text)
    tokens = jieba.lcut(text)
    return [tok.strip() for tok in tokens if tok.strip()]

def initialize_system():
    """
    初始化系统组件，只在首次运行时执行:
    - 加载 LLM & Embeddings
    - 加载向量索引 & 构造检索器
    - 加载 Workflow DAG
    """
    print("Initializing system components...")

    # 初始化LLM
    local_llm_name = 'qwen2.5:14b'
    llm = ChatOllama(model=local_llm_name, temperature=0.7)
    llm_json_mode = ChatOllama(model=local_llm_name, temperature=0.7, format='json')

    # 初始化Embeddings
    embedding_model_name = "bge-m3:latest"
    embedding = OllamaEmbeddings(model=embedding_model_name)

    # 加载本地稀疏索引 & 密集向量库
    sparse_index_path = "./vectordb/bm25_index.pkl"
    dense_faiss_dir = "./vectordb/faiss_index"

    with open(sparse_index_path, "rb") as f:
        sparse_retriever = pickle.load(f)

    vectorstore = FAISS.load_local(dense_faiss_dir, embedding, allow_dangerous_deserialization=True)
    dense_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    # 初始化重排序模型
    reranker_model_name = "/mnt/workspace/pretrain_model/BAAI/bge-reranker-v2-minicpm-layerwise/"
    reranker = LayerWiseFlagLLMReranker(reranker_model_name, use_fp16=True)

    # 可能的外部搜索工具（如 web 搜索）
    web_search_tool = TavilySearchResults(k=3)

    # 从项目代码中导入 build_workflow
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.append(project_root)

    workflow_graph = build_workflow(llm, llm_json_mode, ensemble_retriever, reranker, web_search_tool)

    # 模拟一些全局 config
    workflow_config = {"configurable": {"thread_id": "2"}}

    return workflow_graph, workflow_config

# ========== (2) 确保在 Session State 中只初始化一次 ==========

def get_global_graph_and_config():
    """
    封装一个函数来获得 graph 和 config.
    若在 st.session_state 中尚未初始化，则执行 initialize_system()
    """
    if "graph" not in st.session_state or "workflow_config" not in st.session_state:
        st.session_state.graph, st.session_state.workflow_config = initialize_system()

    return st.session_state.graph, st.session_state.workflow_config

# ========== (3) 其余对话逻辑 ==========

Conversation = Dict
Message = Dict[str, str]

class ConversationManager:    
    def __init__(self):
        if "conversations" not in st.session_state:
            st.session_state.conversations: List[Conversation] = []
            
        if "current_conv_id" not in st.session_state:
            st.session_state.current_conv_id: Optional[int] = None
    
    def start_new_conversation(self) -> None:
        new_id = len(st.session_state.conversations)
        new_conv = {
            "id": new_id,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": [],
            "reasoning": ""
        }
        st.session_state.conversations.append(new_conv)
        st.session_state.current_conv_id = new_id
    
    def get_current_conversation(self) -> Optional[Conversation]:
        if st.session_state.current_conv_id is None:
            return None
        return st.session_state.conversations[st.session_state.current_conv_id]
    
    def add_message(self, role: str, content: str) -> None:
        conv = self.get_current_conversation()
        if conv is not None:
            conv["messages"].append({"role": role, "content": content})
    
    def update_reasoning(self, reasoning: str) -> None:
        conv = self.get_current_conversation()
        if conv is not None:
            conv["reasoning"] += reasoning


def generate_response(messages: List[Dict], max_retries: int) -> Tuple[str, str]:
    """
    将用户/系统消息传给 workflow_graph，
    获取最终输出 + 过程日志
    """
    # 确保我们的 graph & config 已经被初始化
    graph, config = get_global_graph_and_config()

    inputs = {
        "messages": [convert_to_langchain_message(m) for m in messages],
        "max_retries": max_retries,
        "log": "ROUTE QUESTION"
    }
    
    reasoning_log = []
    final_answer = ""
    
    for event in graph.stream(inputs, config, stream_mode="values"):
        if "final_answer" in event and event["final_answer"]:
            final_answer = event["final_answer"]
        
        log_entry = process_event_log(event)
        reasoning_log.append(log_entry)

    return final_answer, "\n".join(reasoning_log)


def convert_to_langchain_message(msg: Message):
    role_mapping = {
        "system":  SystemMessage,
        "user":    HumanMessage,
        "assistant": AIMessage,
        "function": FunctionMessage
    }
    return role_mapping[msg["role"]](content=msg["content"])


def process_event_log(event: Dict) -> str:
    """
    将 event 中的日志内容拼成字符串，方便存入 reason_log
    """
    log_parts = [f"{event['log']}..."] if 'log' in event else []
    
    for msg in event.get('messages', []):
        if isinstance(msg, HumanMessage):
            log_parts.append(f"用户消息: {msg.content}")
        elif isinstance(msg, AIMessage):
            log_parts.append(f"助手消息: {msg.content}")
        elif isinstance(msg, FunctionMessage):
            log_parts.append(f"函数调用: {msg.content}")
    
    return "\n".join(log_parts)


def render_sidebar(conv_manager: ConversationManager):
    with st.sidebar:
        st.markdown("### 🛠️ 参数设置")
        max_retries = st.slider("最大重试次数", 1, 10, 3)
        
        st.markdown("---")
        if st.button("🆕 开启新对话", use_container_width=True):
            conv_manager.start_new_conversation()
            st.rerun()
        
        render_conversation_history(conv_manager)

    # 也可以将 max_retries 返回存到 session_state
    st.session_state.max_retries = max_retries


def render_conversation_history(conv_manager: ConversationManager):
    st.markdown("### 📚 历史对话")
    if not st.session_state.conversations:
        st.write("暂无历史对话")
        return
    
    for conv in reversed(st.session_state.conversations):
        with st.expander(format_conversation_title(conv), expanded=False):
            render_single_conversation(conv)


def format_conversation_title(conv: Conversation) -> str:
    first_question = next(
        (msg["content"] for msg in conv["messages"] if msg["role"] == "user"),
        "新对话"
    )
    return f"{conv['id']+1}. {conv['start_time']} - {truncate_text(first_question)}"


def truncate_text(text: str, length: int = 20) -> str:
    return text[:length] + "..." if len(text) > length else text


def render_single_conversation(conv: Conversation):
    for msg in conv["messages"]:
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f"{role_icon} **{msg['role']}**: {msg['content']}")


def render_main_interface(conv_manager: ConversationManager):
    st.title("DAG+RAG智能问答系统")
    
    current_conv = conv_manager.get_current_conversation()
    if current_conv is None:
        st.info("当前没有活动对话，请点击侧边栏按钮开启新对话。")
        return
    
    # 输出已有消息
    for msg in current_conv["messages"]:
        render_message_bubble(msg, current_conv)

    # 聊天输入框
    if user_input := st.chat_input("请输入您的问题..."):
        conv_manager.add_message("user", user_input)
        
        # 拼接给 DAG 的消息: system + history
        system_msg = {"role": "system", "content": "You are Jarvis, created by Baoxin Wang. You are a helpful assistant."}
        full_history = [system_msg] + current_conv["messages"]

        with st.spinner("正在生成回答..."):
            # 从 session_state 获取重试次数
            max_retries = st.session_state.get("max_retries", 3)

            # 调用 DAG
            answer, reasoning = generate_response(full_history, max_retries=max_retries)
        
        conv_manager.add_message("assistant", answer)
        conv_manager.update_reasoning(reasoning)

        # 触发前端界面更新
        st.rerun()


def render_message_bubble(msg: Message, conversation: Conversation):
    """
    根据角色 (user/assistant) 显示聊天气泡
    如果是 assistant 消息，还可显示“查看推理过程”
    """
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            # 在这里显示所有 reasoning
            if conversation.get("reasoning"):
                with st.expander("查看推理过程"):
                    st.text(conversation["reasoning"])


def main():
    # 确保 graph 只初始化一次
    get_global_graph_and_config()

    # 初始化或获取对话管理器
    conv_manager = ConversationManager()
    if conv_manager.get_current_conversation() is None:
        conv_manager.start_new_conversation()

    # 渲染侧边栏 & 主界面
    render_sidebar(conv_manager)
    render_main_interface(conv_manager)


if __name__ == "__main__":
    main()
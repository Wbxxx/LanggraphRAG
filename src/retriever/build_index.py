# build_index.py

import os
import re
import json
import jieba
import pickle
from tqdm import tqdm
from typing import List

# LangChain 相关
from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 稀疏检索 (BM25)
from langchain_community.retrievers import BM25Retriever

# 密集向量库 (FAISS) + Embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def custom_preprocess_func(text: str) -> List[str]:
    """
    自定义文本预处理函数，用于 BM25:
    1. 去除非中英文及数字下划线的字符
    2. jieba 分词
    """
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_]+", " ", text)  # 移除特殊字符
    tokens = jieba.lcut(text)  # 使用 jieba 进行分词
    return [tok.strip() for tok in tokens if tok.strip()]

def load_txt_documents(folder_path: str) -> List[Document]:
    """
    从指定文件夹下加载所有 .txt 文件，并返回一个 Document 列表
    """
    docs = []
    for filename in tqdm(os.listdir(folder_path), desc="Loading files"):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())
    return docs

def split_docs(
    docs: List[Document], 
    save_path: str, 
    chunk_size=512, 
    chunk_overlap=32
) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 对文档进行切分
    并保存切分后的数据到 jsonl 文件中
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    all_splits = []
    for doc_id, doc in enumerate(docs):
        split_docs = splitter.split_documents([doc])
        for i, split in enumerate(split_docs):
            split.metadata["original_doc_id"] = doc_id  # 记录原始文档 ID
            split.metadata["chunk_index"] = i  # 记录当前块的索引
            split.metadata["total_chunks"] = len(split_docs)  # 当前文档总块数
        all_splits.extend(split_docs)

    with open(save_path, "w", encoding="utf-8") as f:
        for split in all_splits:
            json.dump({
                "page_content": split.page_content,
                "metadata": split.metadata
            }, f, ensure_ascii=False)
            f.write("\n")
    
    print(f"✅ 文档切片已保存至 {save_path}")
    return all_splits

def build_and_save_sparse_retriever(
    doc_splits: List[Document], 
    save_path: str, 
    k=5, 
    rebuild=False
):
    """
    构建并保存 BM25 Retriever
    """
    if not rebuild and os.path.exists(save_path):
        print(f"加载已有 BM25 索引文件: {save_path}")
        return

    print("正在构建 BM25 索引...")
    bm25_params = {"k1": 1.5, "b": 0.75}  # 适用于中文的参数
    sparse_retriever = BM25Retriever.from_documents(
        documents=doc_splits,
        bm25_params=bm25_params,
        preprocess_func=custom_preprocess_func,
        k=k
    )
    
    with open(save_path, "wb") as f:
        pickle.dump(sparse_retriever, f)
    print(f"✅ BM25 索引已存储至 {save_path}")

def build_and_save_dense_vectorstore(
    doc_splits: List[Document], 
    save_dir: str,
    model_name: str = "/mnt/workspace/pretrain_model/BAAI/bge-m3"
):
    """
    构建并保存 FAISS 向量索引
    """
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': False}
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    vectorstore = FAISS.from_documents(doc_splits, embedding)
    vectorstore.save_local(save_dir)
    print(f"✅ FAISS 向量库已存储至 {save_dir}.")

def vdb_store():
    """
    主流程:
    1. 加载文档
    2. 文本切分
    3. 构建并保存 BM25
    4. 构建并保存 FAISS
    """
    folder_path = "MedQA/data/textbooks/zh_paragraph"
    
    os.makedirs("./vectordb", exist_ok=True)
    os.makedirs(dense_faiss_path, exist_ok=True)
    
    split_store_path = "./vectordb/split_docs.jsonl"
    sparse_index_path = "./vectordb/bm25_index.pkl"
    dense_faiss_path = "./vectordb/faiss_index"

    # 1. 加载 .txt 文档
    docs = load_txt_documents(folder_path)
    print(f"Loaded {len(docs)} raw documents.")

    # 2. 文本切分
    doc_splits = split_docs(docs, split_store_path)
    print(f"Split into {len(doc_splits)} chunks.")

    # 3. 构建稀疏检索器 (BM25) + 保存
    build_and_save_sparse_retriever(doc_splits, sparse_index_path)

    # 4. 构建密集向量库 (FAISS) + 保存
    build_and_save_dense_vectorstore(doc_splits, dense_faiss_path)

    print("All done!")

if __name__ == "__main__":
    vdb_store()
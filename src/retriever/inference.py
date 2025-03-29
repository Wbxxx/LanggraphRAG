import pickle
from typing import List

from langchain.docstore.document import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from FlagEmbedding import LayerWiseFlagLLMReranker

def infer():
    """
    1. 加载本地稀疏索引和密集向量库
    2. 组装合奏检索器 (EnsembleRetriever)
    3. 调用重排序模型
    4. 输出检索结果
    """
    # 指定相关路径
    sparse_index_path = "./vectordb/bm25_index.pkl"
    dense_faiss_dir = "./vectordb/faiss_index"

    # 加载稀疏检索器 (BM25)
    with open(sparse_index_path, "rb") as f:
        sparse_retriever = pickle.load(f)

    # 加载密集向量库 (FAISS)
    model_name="/mnt/workspace/pretrain_model/BAAI/bge-m3"
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embedding = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    vectorstore = FAISS.load_local(
        dense_faiss_dir, 
        embedding,
        allow_dangerous_deserialization=True  # 如果 Python/Faiss 版本不兼容，需要此参数
    )
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 5}
    )

    # 合奏检索器
    ensemble_retriever = EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    # 重排序模型
    reranker_model_name = "/mnt/workspace/pretrain_model/BAAI/bge-reranker-v2-minicpm-layerwise/"
    reranker = LayerWiseFlagLLMReranker(reranker_model_name, use_fp16=True)

    # 测试问题
    question = "冬季与初春老年人容易的肺炎，得病后的症状是什么样子？应该如何治疗？"

    # 1) 使用合奏检索器进行初步检索
    docs = ensemble_retriever.invoke(question)

    # 2) 去重
    seen_texts = set()
    unique_docs = []
    for doc in docs:
        text = doc.page_content.strip()
        if text not in seen_texts:
            seen_texts.add(text)
            unique_docs.append(doc)

    # 3) Rerank
    sentence_pairs = [[question, doc.page_content.strip()] for doc in unique_docs]
    scores = reranker.compute_score(
        sentence_pairs,
        batch_size=128,
        max_length=512,
        normalize=True,
        cutoff_layers=[28]
    )

    # 4) 按得分排序
    for doc, score in zip(unique_docs, scores):
        doc.metadata["score"] = score
    sorted_results = sorted(unique_docs, key=lambda x: x.metadata.get("score", 0), reverse=True)

    # 打印结果示例
    for idx, d in enumerate(sorted_results[:5], start=1):
        print(f"Top {idx} | Score: {d.metadata['score']:.4f}")
        print(d.page_content, "\n")

if __name__ == "__main__":
    infer()
import json
from langchain.schema import Document
from langchain_core.messages import HumanMessage, SystemMessage, FunctionMessage
from langgraph.graph import END

lang = "zh"  # 或者 "en"

if lang == "zh":
    from src.dag.prompts import (
        rag_prompt_zh as rag_prompt,
        doc_grader_instructions_zh as doc_grader_instructions,
        doc_grader_prompt_zh as doc_grader_prompt,
        hallucination_grader_instructions_zh as hallucination_grader_instructions,
        hallucination_grader_prompt_zh as hallucination_grader_prompt,
        answer_grader_instructions_zh as answer_grader_instructions,
        answer_grader_prompt_zh as answer_grader_prompt,
    )
elif lang == "en":
    from src.dag.prompts import (
        rag_prompt_en as rag_prompt,
        doc_grader_instructions_en as doc_grader_instructions,
        doc_grader_prompt_en as doc_grader_prompt,
        hallucination_grader_instructions_en as hallucination_grader_instructions,
        hallucination_grader_prompt_en as hallucination_grader_prompt,
        answer_grader_instructions_en as answer_grader_instructions,
        answer_grader_prompt_en as answer_grader_prompt,
    )
else:
    raise ValueError("Unsupported language. Please set 'lang' to 'zh' or 'en'.")

def format_docs(docs):
    """将所有 docs 的内容拼接起来"""
    return "\n\n".join(doc.page_content for doc in docs)

def format_documents(documents):
    func_documents = ""
    for doc in documents:
        func_documents += "--" * 10
        func_documents += doc.page_content + "\n"
    return func_documents

def extract_question(messages):
    for msg in reversed(messages): 
        if isinstance(msg, HumanMessage):
            question = msg.content
            break
    return question

def retrieve(state, retriever):
    """ 检索节点：从 vectorstore 中检索文档 """
    print("---RETRIEVE---")
    messages = state["messages"]
    question = extract_question(messages)
    documents = retriever.invoke(question)
    return {"question": question, "documents": documents, "log": "RETRIEVE"}

def rerank(state, reranker):
    """ 检索节点：从 vectorstore 中检索文档 """
    print("---RERANK---")
    question = state["question"]
    documents = state["documents"]
    messages = state["messages"]
    seen_texts = set()
    unique_docs = []
    for doc in documents:
        text = doc.page_content.strip()
        if text not in seen_texts:
            seen_texts.add(text)
            unique_docs.append(doc)

    sentence_pairs = [[question, doc.page_content.strip()] for doc in unique_docs]
    scores = reranker.compute_score(
        sentence_pairs,
        batch_size=128,
        max_length=512,
        normalize=True,
        cutoff_layers=[28]
    )
    for doc, score in zip(unique_docs, scores):
        doc.metadata["score"] = score
    sorted_results = sorted(unique_docs, key=lambda x: x.metadata.get("score", 0), reverse=True)
    func_documents = format_documents(sorted_results)
    messages.append(FunctionMessage(name="retrieve&rerank", content=func_documents))
    
    return {"messages": messages, "documents": sorted_results, "log": "RERANK"}

def generate(state, llm):
    """ 生成节点：使用 RAG 进行回答生成 """
    print("---GENERATE---")
    question = state["question"]
    messages = state["messages"]
    documents = state["documents"]
    loop_step = state.get("loop_step", 0)

    # RAG generation
    docs_txt = format_docs(documents)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    final_answer = generation.content
    messages.append(generation)
    return {"final_answer": final_answer, "messages": messages, "loop_step": loop_step + 1, "log": "GENERATE"}

def general_conversation(state, llm):
    print("---GENERAL CONVERSATION---")
    messages = state["messages"]
    messages.append(llm.invoke(messages))
    final_answer = messages[-1].content
    return {"final_answer": final_answer, "messages": messages,"log": "GENERAL CONVERSATION"}

def grade_documents(state, llm_json_mode):
    """
    打分节点：判断检索到的文档是否与问题相关
    不相关的文档会触发后续的 web_search
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    messages = state["messages"]

    filtered_docs = []
    grade_scores = []
    web_search = "No"
    for d in documents:
        doc_grader_prompt_formatted = doc_grader_prompt.format(document=d.page_content, question=question)
        result = llm_json_mode.invoke(
            [SystemMessage(content=doc_grader_instructions), HumanMessage(content=doc_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)['binary_score']
        grade_scores.append(grade)
        # 如果相关
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
    messages.append(FunctionMessage(name="grade_documents", content=" ".join(grade_scores)))

    return {"messages":messages, "documents": filtered_docs, "web_search": web_search, "log": "CHECK DOCUMENT"}

def web_search(state, web_search_tool):
    """
    Web 搜索节点：如果文档不够相关，则调用 Web 搜索
    """
    print("---WEB SEARCH---")
    
    messages = state["messages"]
    question = extract_question(messages)
    documents = state.get("documents", [])

    docs = web_search_tool.invoke({"query": question})
    url_list = "\n".join([d["url"] for d in docs])
    web_results = "\n".join([d["content"] for d in docs])
    documents.append(Document(metadata={"source": url_list},page_content=web_results))
    func_documents = format_documents(documents)
    messages.append(FunctionMessage(name="web_search", content=func_documents))
    return {"messages": messages, "question": question, "documents": documents, "log": "WEB SEARCH"}

def grade_generation_v_documents_and_question(state, llm_json_mode):
    """
    判断生成是否符合文档内容（无幻觉）且回答了问题
    """
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    messages = state["messages"]
    max_retries = state.get("max_retries", 3)

    # 判断是否有幻觉
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=format_docs(documents),
        generation=messages[-1].content
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions), 
         HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    grade = json.loads(result.content)['binary_score']

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # 检查回答是否回答了问题
        answer_grader_prompt_formatted = answer_grader_prompt.format(
            question=question, generation=messages[-1].content
        )
        result = llm_json_mode.invoke(
            [SystemMessage(content=answer_grader_instructions), HumanMessage(content=answer_grader_prompt_formatted)]
        )
        grade = json.loads(result.content)['binary_score']
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            state["log"] = "DECISION: GENERATION ADDRESSES QUESTION"
            return "useful"
        elif state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            state["log"] = "DECISION: GENERATION DOES NOT ADDRESS QUESTION"
            return "not useful"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            state["log"] = "DECISION: MAX RETRIES REACHED"
            return "max retries"
    else:
        # 有幻觉
        if state["loop_step"] <= max_retries:
            print("---DECISION: GENERATION IS NOT GROUNDED, RE-TRY---")
            state["log"] = "DECISION: GENERATION IS NOT GROUNDED, RE-TRY"
            return "not supported"
        else:
            print("---DECISION: MAX RETRIES REACHED---")
            state["log"] = "DECISION: MAX RETRIES REACHED"
            return "max retries"
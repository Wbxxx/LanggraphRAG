import operator
from typing_extensions import TypedDict
from typing import List, Annotated

from langchain.schema import Document
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.dag.nodes import (
    retrieve,
    rerank,
    generate,
    grade_documents,
    web_search,
    general_conversation,
    grade_generation_v_documents_and_question
)

from src.dag.edges import route_question, decide_to_generate


class GraphState(TypedDict):
    messages: list
    question: str
    final_answer: str
    web_search : str 
    max_retries : int  
    answers : int 
    loop_step: Annotated[int, operator.add] 
    documents : List[str] 
    log : str 

def build_workflow(llm, llm_json_mode, ensemble_retriever, reranker, web_search_tool):
    memory = MemorySaver()
    workflow = StateGraph(GraphState)

    # 添加节点
    workflow.add_node("websearch", lambda state: web_search(state, web_search_tool))
    workflow.add_node("conversation", lambda state: general_conversation(state, llm))
    workflow.add_node("retrieve", lambda state: retrieve(state, ensemble_retriever)) 
    workflow.add_node("rerank", lambda state: rerank(state, reranker))
    workflow.add_node("grade_documents", lambda state: grade_documents(state, llm_json_mode))
    workflow.add_node("generate", lambda state: generate(state, llm))

    # 条件入口
    workflow.set_conditional_entry_point(
        lambda state: route_question(state, llm_json_mode),
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
            "conversation": "conversation"
        }
    )

    workflow.add_edge("conversation", END)
    workflow.add_edge("websearch", "generate")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )

    workflow.add_conditional_edges(
        "generate",
        lambda state: grade_generation_v_documents_and_question(state, llm_json_mode),
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
            "max retries": END,
        },
    )

    return workflow.compile(checkpointer=memory)
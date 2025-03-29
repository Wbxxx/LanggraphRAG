import json
from langchain_core.messages import HumanMessage, SystemMessage

lang = "zh"
if lang == "zh":
    from src.dag.prompts import router_instructions_zh as router_instructions
elif lang == "en":
    from src.dag.prompts import router_instructions_en as router_instructions
else:
    raise ValueError("Unsupported language. Please set 'lang' to 'zh' or 'en'.")

def route_question(state, llm_json_mode):
    """
    根据用户提问，决定调用 websearch 还是 vectorstore
    """
    print("---ROUTE QUESTION---")
    route_messages = [SystemMessage(content=router_instructions)] + state["messages"]
    route_question_result = llm_json_mode.invoke(route_messages)
    source = json.loads(route_question_result.content)['datasource']
    print(source)
    if source == 'web search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    elif source == 'general conversation':
        print("---ROUTE QUESTION TO CONVERSATION---")
        return 'conversation'

def decide_to_generate(state):
    """
    如果打分后有不相关文档，则继续 websearch，否则直接 generate
    """
    print("---ASSESS GRADED DOCUMENTS---")
    web_search = state["web_search"]
    if web_search == "Yes":
        print("---DECISION: WE WILL WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"
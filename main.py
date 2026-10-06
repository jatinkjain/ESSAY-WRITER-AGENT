from langgraph.graph import StateGraph, START,END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage,HumanMessage,SystemMessage,AIMessage,ChatMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated,List
from pydantic import BaseModel
from tavily import TavilyClient
from dotenv import load_dotenv
import os
import operator

load_dotenv()

class Agentstate(TypedDict):
    task: str
    lnode: str
    plan: str
    draft: str
    critique: str
    Content: list[str]
    queries: list[str]
    revision_number: int
    max_revisions: int
    count: Annotated[int,operator.add]

class Queries(BaseModel):
    queries: list[str]

class essaywriter():
    def __init__(self):
        self.model = ChatOpenAI(model="", temperature=0)
        self.PLAN_PROMPT = ("You are an expert writer tasked with writing a high level outline of a short 3 paragraph essay. "
                            "Write such an outline for the user provided topic. Give the three main headers of an outline of "
                             "the essay along with any relevant notes or instructions for the sections. ")
        self.WRITER_PROMPT = ("You are an essay assistant tasked with writing excellent 3 paragraph essays. "
                              "Generate the best essay possible for the user's request and the initial outline. "
                              "If the user provides critique, respond with a revised version of your previous attempts. "
                              "Utilize all the information below as needed: \n"
                              "------\n"
                              "{content}")
        self.RESEARCH_PLAN_PROMPT = ("You are a researcher charged with providing information that can "
                                     "be used when writing the following essay. Generate a list of search "
                                     "queries that will gather "
                                     "any relevant information. Only generate 3 queries max.")
        self.REFLECTION_PROMPT = ("You are a teacher grading an 3 paragraph essay submission. "
                                  "Generate critique and recommendations for the user's submission. "
                                  "Provide detailed recommendations, including requests for length, depth, style, etc.")
        self.RESEARCH_CRITIQUE_PROMPT = ("You are a researcher charged with providing information that can "
                                         "be used when making any requested revisions (as outlined below). "
                                         "Generate a list of search queries that will gather any relevant information. "
                                         "Only generate 2 queries max.")        
        self.tavily = TavilyClient(api_key=[])
        graph = StateGraph(Agentstate)
        graph.add_node("planner",self.plan_node)
        graph.add_node("research_plan",self.research_plan_node)
        graph.add_node("generate",self.generation_node)
        graph.add_node("reflect",self.reflection_node)
        graph.add_node("research_critique",self.research_critique_node)
        graph.add_conditional_edges("generate",self.should_continue,{END:END,"reflect":"reflect"})
        graph.add_edge(START,"planner")
        graph.add_edge("planner","research_plan")
        graph.add_edge("research_plan","generate")
        graph.add_edge("reflect","research_critique")
        graph.add_edge("research_critique","generate")
        self.abot = graph.compile()

    def plan_node(self,state:Agentstate):
        messages = [
            SystemMessage(content= self.PLAN_PROMPT),
            HumanMessage(content= state["task"])
        ]

        response = self.model.invoke(messages)
        return{"plan": response.content,"lnode":"planner","count":1}

    def research_plan_node(self, state:Agentstate):
        messages = [
            SystemMessage(content= self.RESEARCH_PLAN_PROMPT),
            HumanMessage(content= state["task"])
        ]
        queries = self.model.with_structured_output(Queries).invoke(messages)

        content = state["Content"] or []
        for q in queries.queries:
            response = self.Tavily.search(query= q, max_results= 2)
            for r in response['results']:
                content.append(r['content'])
            return{"content": content, "lnode": "research_plan", "queries": queries.queries, "count": 1}
            
    def generation_node(self, state:Agentstate):
        content = "\n\n".join(state["Content"] or [])
        messages = [
            SystemMessage(content= self.WRITER_PROMPT.format(content= content)),
            HumanMessage(content=f"{state['task']} \n\n here i my plan: \n\n{state['plan']}")
        ]
        response= self.model.invoke(messages)
        return{"draft":response.content, "lnode":"generate", "count": 1, "revision_number":state.get("revision_number",1)+1 }
    
    def reflection_node(self,state:Agentstate):
        messages = [
            SystemMessage(content= self.REFLECTION_PROMPT),
            HumanMessage(content= state["draft"])
        ]
        response = self.model.invoke(messages)
        return{"critique":response.content, "lnode":"reflect", "count":1}

    def research_critique_node(self,state:Agentstate):
        queries = self.model.with_structured_output(Queries).invoke([
            SystemMessage(content= self.RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content= state["critique"])
        ])
        content = state["Content"] or []
        for q in queries.queries:
            response = self.tavily.search(query= q, max_results=2)
            for r in response['results']:
                content.append(r['content'])
        return{"content":content, "lnode":"research_critique", "count":1}        

    def should_continue(self,state:Agentstate):
        if state["revision_number"] > state["max_revisions"]:
            return END 
        return "reflect"


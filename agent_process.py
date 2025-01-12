from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from typing import List, Dict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
import json
import re
from datetime import datetime
from langchain_openai import ChatOpenAI
from prompts import query_agent_prompt, orchestrator_agent_prompt, human_handoff_agent_prompt
from schemas import query_agent_schema, orchestrator_agent_schema
from langchain_core.messages import HumanMessage, AIMessage
import sys
import os
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
import config

os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY

llm_70b = ChatGroq(
    model="llama-3.3-70b-versatile"
)
llm_8b = ChatGroq(
    model="gemma2-9b-it"
)
llm_openai = ChatOpenAI(
    model = "gpt-4o-mini"
)


def agent_process(prompt, chat_history):
    # Start timing the process
    start_time = datetime.now()

    # Execute the query agent and get the result
    query_agent_result, language = query_agent(llm=llm_70b, prompt=prompt, system_prompt=query_agent_prompt,chat_history=chat_history)
    print(f'\n\nQuery Agent result...\n{query_agent_result}')
    if query_agent_result['greeting'] != '':
        
        # Translate the greeting if present
        translate_result, language_2 = translate_text(language, query_agent_result['greeting'])
        print(f"\n{translate_result}")
        latest_chat_history = [HumanMessage(content=prompt), AIMessage(content=translate_result)]
        chat_history.extend(latest_chat_history)
        end_time = datetime.now()
        print("\n\nduration for query", end_time - start_time)
        return translate_result

    # Execute the orchestrator agent
    orchestrator_agent_result = orchestrator_agent(llm=llm_70b, prompt=query_agent_result['action'], system_prompt=orchestrator_agent_prompt,chat_history=chat_history)
    print(f'\n\nOrchestrator Agent result...\n{orchestrator_agent_result}')

    if orchestrator_agent_result["agent_to_call"] == "human_handoff_agent":
        # Handle human handoff if required
        human_handoff_agent_result = human_handoff_agent(llm=llm_70b, prompt=orchestrator_agent_result["handoff_information"], system_prompt=human_handoff_agent_prompt,chat_history=chat_history)
        
        translate_result, language_2 = translate_text(language, human_handoff_agent_result.content)
        print(f'\n\nHuman handoff Agent result...\n{translate_result}')

        latest_chat_history = [HumanMessage(content=prompt), AIMessage(content=translate_result)]
        chat_history.extend(latest_chat_history)
        end_time = datetime.now()
        print("\n\nduration for query", end_time - start_time)
        return translate_result
    
    # Execute the document agent
    document_agent_result = document_agent(prompt)
    print(f'\n\nDocument Agent result...\n{document_agent_result}')
    if document_agent_result["agent_to_call"] == "human_handoff_agent":
        # Handle human handoff if required
        human_handoff_agent_result = human_handoff_agent(llm=llm_8b, prompt=document_agent_result["content"], system_prompt=human_handoff_agent_prompt, chat_history=chat_history)
        translate_result, language_2 = translate_text(language, human_handoff_agent_result.content)

        latest_chat_history = [HumanMessage(content=prompt), AIMessage(content=translate_result)]
        chat_history.extend(latest_chat_history)

        print(f'\n\nHuman handoff Agent result...\n{translate_result}')
        end_time = datetime.now()
        print("\n\nduration for query", end_time - start_time)
        return translate_result
    
    # Execute the response agent
    response_agent_result = response_agent(llm=llm_8b, prompt=query_agent_result['action'],context=document_agent_result["content"])
    print(f'\n\nResponse Agent result...\n{response_agent_result}')
    translate_result, language_2 = translate_text(language, response_agent_result)

    latest_chat_history = [HumanMessage(content=prompt), AIMessage(content=translate_result)]
    chat_history.extend(latest_chat_history)

    end_time = datetime.now()
    print("\n\nduration for query", end_time - start_time)
    return translate_result






def query_agent(llm, prompt, system_prompt, chat_history):
    translated_prompt, language = translate_text('en', prompt)
    return agent_executor(llm, translated_prompt, system_prompt, chat_history, schema=query_agent_schema), language
def orchestrator_agent(llm, prompt, system_prompt, chat_history):
    return agent_executor(llm, prompt, system_prompt, chat_history, schema=orchestrator_agent_schema)
    
def human_handoff_agent(llm, prompt, system_prompt,chat_history):
    return agent_executor(llm, prompt, system_prompt, chat_history)

def document_agent(prompt):
    # Set the OpenAI API key
    openai.api_key = os.environ['OPENAI_API_KEY']
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Specify which embedding model

    # Connect to the Pinecone index using LangChain's Pinecone wrapper
    pinecone_index_name = "real-estate-docs"
    vector_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings, namespace="circulars")

    # Perform similarity search
    results = vector_store.similarity_search_with_score(query=prompt,k=3)
    context_list = []
    total_similarity_score = 0
    for doc, score in results:
        context_list.append(doc.page_content)
        total_similarity_score += score
    
    # Calculate average similarity score
    average_similarity_score = total_similarity_score/3
    print("\n\nsimilarity_score is: ", average_similarity_score)
    if average_similarity_score < 0.70:
        # Escalate if similarity score is too low
        return {"agent_to_call":"human_handoff_agent", "content": "There were no good matches after retrieving. Requires escalation"}
    
    # Compile context from retrieved documents
    context = "\n\n".join(context_list)
    return {"agent_to_call":"response_agent", "content": context}

def response_agent(llm, prompt, context):
    graph = Neo4jGraph(url=config.NEO4J_URI, username=config.NEO4J_USERNAME, password=config.NEO4J_PASSWORD)
    chain = GraphCypherQAChain.from_llm(llm_openai, graph=graph, verbose=True, allow_dangerous_requests=True)

    relations = chain.invoke({"query": prompt})

    prompt_template = PromptTemplate(
    template="""
    Use the following context to answer the question as accurately as possible:
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
    )
    chain = prompt_template | llm | StrOutputParser()

   

    return chain.invoke({"context": context + " " + relations['result'], "question": prompt})

def agent_executor(llm, prompt, system_prompt, chat_history=None, schema=""):
    if chat_history is None:
        chat_history = []
    template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
    # return llm.invoke("input":template)
    if schema == "":
        chain = template | llm
    else:
        structured_llm = llm.with_structured_output(schema)
        chain = template | structured_llm

    
    try:
        print("\nAgent invoking...")
        response = chain.invoke({
            "input": prompt,
            "chat_history": chat_history
        })

        return response
    except Exception as e:
        print(f"Agent Error: {str(e)}")



def translate_text(target_language: str, text: str):
    
    from google.cloud import translate_v2

    translate_client = translate_v2.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    detection = translate_client.detect_language(text)
    source_lang = detection["language"]

    if source_lang != target_language:
        print("\n\nTranslating")
        translation = translate_client.translate(
            text, target_language=target_language
        )
        result =  translation["translatedText"]

    else:
        result = text
    return result, source_lang

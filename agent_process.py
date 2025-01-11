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
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import config

os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY

llm_70b = ChatGroq(
    model="llama-3.3-70b-versatile"
)
llm_8b = ChatGroq(
    model="llama3-8b-8192"
)

def agent_process(prompt, chat_history):
    start_time = datetime.now()

    query_agent_result, language = query_agent(llm=llm_70b, prompt=prompt, system_prompt=query_agent_prompt,chat_history=chat_history)
    print(f'\n\nQuery Agent result...\n{query_agent_result}')
    if query_agent_result['greeting'] != '':
        
        translate_result, language_2 = translate_text(language, query_agent_result['greeting'])
        print(f"\n{translate_result}")
        latest_chat_history = [HumanMessage(content=prompt), AIMessage(content=translate_result)]
        chat_history.extend(latest_chat_history)
        end_time = datetime.now()
        print("\n\nduration for query", end_time - start_time)
        return translate_result

    orchestrator_agent_result = orchestrator_agent(llm=llm_70b, prompt=query_agent_result['action'], system_prompt=orchestrator_agent_prompt,chat_history=chat_history)
    print(f'\n\nOrchestrator Agent result...\n{orchestrator_agent_result}')

    if orchestrator_agent_result["agent_to_call"] == "human_handoff_agent":
        human_handoff_agent_result = human_handoff_agent(llm=llm_8b, prompt=orchestrator_agent_result["handoff_information"], system_prompt=human_handoff_agent_prompt,chat_history=chat_history)
        
        translate_result, language_2 = translate_text(language, human_handoff_agent_result.content)
        print(f'\n\nHuman handoff Agent result...\n{translate_result}')

        latest_chat_history = [HumanMessage(content=prompt), AIMessage(content=translate_result)]
        chat_history.extend(latest_chat_history)
        end_time = datetime.now()
        print("\n\nduration for query", end_time - start_time)
        return translate_result
    
    document_agent_result = document_agent(prompt)
    print(f'\n\nDocument Agent result...\n{document_agent_result}')
    if document_agent_result["agent_to_call"] == "human_handoff_agent":
        human_handoff_agent_result = human_handoff_agent(llm=llm_8b, prompt=document_agent_result["content"], system_prompt=human_handoff_agent_prompt)
        translate_result, language_2 = translate_text(language, human_handoff_agent_result.content)

        latest_chat_history = [HumanMessage(content=prompt), AIMessage(content=translate_result)]
        chat_history.extend(latest_chat_history)

        print(f'\n\nHuman handoff Agent result...\n{translate_result}')
        end_time = datetime.now()
        print("\n\nduration for query", end_time - start_time)
        return translate_result
    
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
    openai.api_key = os.environ['OPENAI_API_KEY']
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Specify which embedding model

    # Connect to the Pinecone index using LangChain's Pinecone wrapper
    pinecone_index_name = "real-estate-docs"
    vector_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)

    # Define the retrieval mechanism
    # retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    results = vector_store.similarity_search_with_score(query=prompt,k=3)
    context_list = []
    total_similarity_score = 0
    for doc, score in results:
        context_list.append(doc.page_content)
        total_similarity_score += score
    
    average_similarity_score = total_similarity_score/3
    print("\n\nsimilarity_score is: ", average_similarity_score)
    if average_similarity_score < 0.70:
        return {"agent_to_call":"human_handoff_agent", "content": "There were no good matches after retrieving. Requires escalation"}
    
    context = "\n\n".join(context_list)
    return {"agent_to_call":"response_agent", "content": context}

def response_agent(llm, prompt, context):
    prompt_template = PromptTemplate(
    template="""
    Use the following context to answer the question as accurately as possible:
    Context: {context}
    Question: {question}
    Answer:""",
    input_variables=["context", "question"]
    )
    chain = prompt_template | llm | StrOutputParser()

    return chain.invoke({"context": context, "question": prompt})

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



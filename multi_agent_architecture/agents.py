from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import List, Dict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
import json
import re
import time
from langchain_openai import ChatOpenAI
from prompts import query_agent_prompt


llm = ChatOpenAI(model="gpt-4-turbo")

def agent_process(prompt, chat_history):
    query_agent_result = query_agent(llm=llm, prompt=prompt, system_prompt=query_agent_prompt,chat_history=chat_history)
    return

def query_agent(llm, prompt, system_prompt, chat_history):
    output = agent_executor(llm, prompt, system_prompt, chat_history)


def agent_executor(llm, prompt, system_prompt, chat_history):
    template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
    chain = template | llm
    
    print("Invoking LLM...")
    try:
        response = chain.invoke({
            "input": prompt,
            "chat_history": chat_history
        })

        return response.content
    except Exception as e:
        print(f"Agent Error: {str(e)}")
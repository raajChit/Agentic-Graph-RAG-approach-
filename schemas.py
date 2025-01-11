query_agent_schema = {
    "title": "query identifier",
    "description": "Will identify few aspects of the user query",
    "type": "object",
    "properties": {
        "greeting": {
            "type": "string",
            "description": "add in the reply to the user if the user is greeting you",
        },
        "action": {
            "type": "str",
            "description": " menion what the user's question is and identify wether it falls under a knowledge base request, sensitive/personal topic or a complex query. ",
        },
    },
    "required": ["greeting", "action"],
}

orchestrator_agent_schema = {
    "title": "orchestrator",
    "description": "Will decide where to route.",
    "type": "object",
    "properties": {
        "agent_to_call": {
            "type": "string",
            "description": "mention which agents needs to be called - document_agent or human_handoff_agent",
        },
        "handoff_information": {
            "type": "string",
            "description": " if the human_handoff_agent is called explain the user request and mention why it requires human handoff",
        },
    },
    "required": ["agent_to_call", "handoff_information"],
}
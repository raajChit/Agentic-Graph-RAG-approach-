import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv("./.env")

# Project configuration variables
PROJECT_ID = os.getenv("PROJECT_ID")  # ID of the project
LOCATION = os.getenv("LOCATION", "us")  # Default location set to 'us'
PROCESSOR_ID = os.getenv("PROCESSOR_ID")  # ID of the processor

# Credentials and API keys
CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")  # Path to Google credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # API key for OpenAI
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # API key for GROQ

# Neo4j database connection details
NEO4J_URI = os.getenv("NEO4J_URI")  # URI for Neo4j database
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")  # Username for Neo4j
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  # Password for Neo4j

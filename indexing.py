from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer

def index_files():
    print("\n\nIndexing....")
    # Initialize OpenAI Embeddings using LangChain
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")  # Specify which embedding model

    # Load all text files from a directory
    directory_path = "./processed_files"  # directory path with all the national weather service documents
    loader = DirectoryLoader(directory_path, glob="*.txt", loader_cls=TextLoader)  # Load only .txt files
    documents = loader.load()

    # Use a TextSplitter to split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_documents = text_splitter.split_documents(documents)

    # Connect to the Pinecone index using LangChain's Pinecone wrapper
    # Add all the split documents into the Pinecone vector database
    pinecone_index_name = "real-estate-docs"
    vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings, namespace="circulars")
    vectorstore.add_documents(documents=split_documents)

    print("\nSUCCESS:Embeddings from text files residing in the directory, created, and inserted in Pinecone Vector Database successfully!")

    # Initialize the language model for graph transformation
    llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini-2024-07-18", api_key=config.OPENAI_API_KEY)

    # Transform documents into graph format
    llm_transformer = LLMGraphTransformer(llm=llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Connect to Neo4j and add graph documents
    graph = Neo4jGraph(url=config.NEO4J_URI, username=config.NEO4J_USERNAME, password=config.NEO4J_PASSWORD)
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )
    print("\nDocuments relationships and hierarchy added to knowledge graph")


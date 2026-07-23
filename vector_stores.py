from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv()
os.getenv('HF_HOME')

from langchain_huggingface import HuggingFaceEmbeddings


embeddings_model = HuggingFaceEmbeddings(
    model_name = 'Qwen/Qwen3-Embedding-0.6B',
    encode_kwargs = {'normalize_embeddings' : True},
    # cache_folder = './cache'
)

#Persistent Directory 
CHROMA_DIR = './DB/chroma_db'

# Sample documents
SAMPLE_DOCS = [
    Document(
        page_content="LangChain is a framework for developing applications powered by language models.",
        metadata={"source": "langchain_docs", "topic": "overview"},
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "topic": "overview"},
    ),
    Document(
        page_content="Vector stores are databases optimized for storing and searching embeddings.",
        metadata={"source": "vector_guide", "topic": "database"},
    ),
    Document(
        page_content="RAG combines retrieval with generation for more accurate LLM responses.",
        metadata={"source": "rag_guide", "topic": "architecture"},
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors for semantic similarity.",
        metadata={"source": "embeddings_guide", "topic": "fundamentals"},
    ),
    Document(
        page_content="Chroma is an open-source embedding database for AI applications.",
        metadata={"source": "chroma_docs", "topic": "database"},
    ),
    Document(
        page_content="FAISS is a library for efficient similarity search developed by Facebook.",
        metadata={"source": "faiss_docs", "topic": "database"},
    ),
    Document(
        page_content="Pinecone is a managed vector database service for production workloads.",
        metadata={"source": "pinecone_docs", "topic": "database"},
    ),
]


def chroma_basics():


    #create store from docs
    vector_store = Chroma.from_documents(
        documents = SAMPLE_DOCS, embedding = embeddings_model, persist_directory = CHROMA_DIR 
    )

    print(f'Vector Store created {vector_store._collection_metadata}\n')

    #perform similarity search
    query = 'What is LangChain?'
    results = vector_store.similarity_search(query, k=2)

    print(f'Top 2 results for query "{query}" :')
    for i, doc in enumerate(results):

        print(f'Result {i+1} : {doc.page_content} (Source: {doc.metadata["source"]})')


def similarity_search_with_scores():


    vector_store = Chroma.from_documents(
        documents = SAMPLE_DOCS, embedding = embeddings_model, persist_directory = CHROMA_DIR
    )

    query = 'Explain vector stores'
    results_with_scores = vector_store.similarity_search_with_score(query, k=3)

    print(f"Top 3 results with scores for query '{query}':")

    for i, (doc, score) in enumerate(results_with_scores):

        print(f'Result {i+1} : {doc.page_content} (Score: {score:.4f}, Source: {doc.metadata["source"]})') 
        # these are distance scores and not similarity scores hence the more close to 0 the more similar the answer is !


def metadata_filtering():

    
    vector_store = Chroma.from_documents(
        documents = SAMPLE_DOCS, embedding = embeddings_model, persist_directory = CHROMA_DIR
    )

    query = 'What databases are available?'

    #without metadata filtering
    results = vector_store.similarity_search(query, k=5)

    print(f"Top 5 results without metadata filtering for query '{query}':")
    for i, doc in enumerate(results):
        print(f'Result {i+1} : {doc.page_content} (Source: {doc.metadata["source"]})')


    #with metadata filtering
    filter_criteria = {'topic' : 'database'}

    filtered_results = vector_store.similarity_search(query, k=5, filter = filter_criteria)

    print(f"Top 5 results with metadata filtering for query '{query}':")
    for i, doc in enumerate(filtered_results):
        print(f'Result {i+1} : {doc.page_content} (Source: {doc.metadata["source"]})')


if __name__ == "__main__":

    # chroma_basics()
    # similarity_search_with_scores()
    metadata_filtering()
    #each time the embedding model will be changed the vector store must be recreated cause it'll give a dimension error due to the previous embeddings model used in it. 
    # and more dimensions means higher accuracy in results as in 384 dimensions the answer was in repeatation and in 1024 dimension the answer is unique  
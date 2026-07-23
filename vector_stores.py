from langchain_chroma import Chroma
import tempfile
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings_model = HuggingFaceEmbeddings(
    model_name = 'sentence-transformers/all-MiniLM-L6-v2',
    encode_kwargs = {'normalize_embeddings' : True}
)

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

    with tempfile.TemporaryDirectory() as tmpdir:

        #create store from docs
        vector_store = Chroma.from_documents(
            documents = SAMPLE_DOCS, embedding = embeddings_model, persist_directory = tmpdir 
        )

        print(f'Vector Store created {vector_store._collection_metadata}\n')

        #perform similarity search
        query = 'What is LangChain?'
        results = vector_store.similarity_search(query, k=2)

        print(f'Top 2 results for query "{query}" :')
        for i, doc in enumerate(results):

            print(f'Result {i+1} : {doc.page_content} (Source: {doc.metadata['source']})')


def similarity_search_with_scores():

    with tempfile.TemporaryDirectory() as tmpdir:

        vector_store = Chroma.from_documents(
            documents = SAMPLE_DOCS, embedding = embeddings_model, persist_directory = tmpdir
        )

        query = 'Explain vector stores'
        results_with_scores = vector_store.similarity_search_with_score(query, k=3)

        print(f"Top 3 results with scores for query '{query}':")

        for i, (doc, score) in enumerate(results_with_scores):

            print(f'Result {i+1} : {doc.page_content} (Score: {score:.4f}, Source: {doc.metadata['source']})') #these are distance scores and not similarity scores


if __name__ == "__main__":

    # chroma_basics()
    similarity_search_with_scores()
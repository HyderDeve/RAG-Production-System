from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model 

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv
import os 

load_dotenv()

os.getenv('HF_HOME')
groq_api_key = os.getenv('GROQ_API_KEY')

from langchain_huggingface import HuggingFaceEmbeddings


embeddings_model = HuggingFaceEmbeddings(
    model_name = 'Qwen/Qwen3-Embedding-0.6B',
    encode_kwargs = {'normalize_embeddings' : True},
)

#Persistent Directory 
CHROMA_DIR = './DB/chroma_db_rag'


# Sample knowledge base
KNOWLEDGE_BASE = """# LangChain Framework

LangChain is a framework for developing applications powered by language models. It was created by Harrison Chase in October 2022.

## Core Components

1. **Models**: LangChain supports various LLM providers including OpenAI, Anthropic, and local models.

2. **Prompts**: Templates for structuring inputs to language models.

3. **Chains**: Sequences of calls to models and other components.

4. **Agents**: Systems that use LLMs to determine which actions to take.

5. **Memory**: Components for persisting state between chain/agent calls.

## LangGraph

LangGraph is a library for building stateful, multi-actor applications. Key features:
- State management
- Cycles and loops
- Human-in-the-loop
- Persistence

## Pricing

LangChain itself is open source and free. LangSmith (the observability platform) has a free tier and paid plans starting at $39/month.

## Getting Started

Install with: pip install langchain langchain-openai
Create your first chain in under 10 lines of code.
"""


def create_kb():

    '''Create a vector store from knowledge base.'''

    # split the knowledge base into chunks
    
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)

    doc = Document(page_content = KNOWLEDGE_BASE, 
                   metadata = {'source' : 'langchain_knowledge_base.md'})

    chunks = splitter.split_documents([doc])

    #create a vector store from chunks

    vector_store = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings_model,
        persist_directory = CHROMA_DIR

    )

    return vector_store


def demo_basic_rag():

    vector_store = create_kb()

    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {'k' : 2})

    llm = init_chat_model(model = 'openai/gpt-oss-120b', temperature = 0.5, model_provider = 'groq')

    # RAG Prompt Template
    
    prompt = ChatPromptTemplate.from_template(
        '''
Answer the question based only on the context:

{context}

Question : {question}

Answer:

Make sure to answer in a concise manner, 
and if you don't know the answer, just say "I do not know". '''
    )

    # Format retrieved docs 
    def format_docs(docs):

        return "\n\n".join([doc.page_content for doc in docs]) #list comprehension

    # RAG Chain
    rag_chain = (
        {'context' : retriever | format_docs, 'question' : RunnablePassthrough()} # the runnable passthrough just says the question remains unchanged
        | prompt
        | llm
        | StrOutputParser()
    )

    #Test the RAG chain

    # Test

    questions = [
        'What is Langchain?',
        'Who created Langchain?',
        'What is LangGraph used for?'
    ]

    print('Basic RAG Demo: \n')
    for q in questions:
        answer = rag_chain.invoke(q)
        print(f'Question: {q}')
        print(f'Answer: {answer}\n')



if __name__ == '__main__':
    
    demo_basic_rag()
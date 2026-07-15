import os
import tempfile
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader, 
    WebBaseLoader, 
    DirectoryLoader, 
    PyPDFLoader
)


from dotenv import load_dotenv

load_dotenv()

def load_text_file():

    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as temp_file:
        temp_file.write(b'Hello, this is a sample text file.')
        temp_file_path = temp_file.name

    try:
        
        #load a text file using TextLoader
        loader = TextLoader(temp_file_path)
        documents = loader.load()

        print(f' Loaded {len(documents)} documents')
        print(f'Content preview: {documents[0].page_content[:100]}...')
        print(f'Metadata: {documents[0].metadata}')

        # for docs in documents:
        #     print('Document: \n')
        #     print(docs)
        #     print(docs.page_content)

    finally:
        # Clean up the temporary file
        os.remove(temp_file_path)

def pdf_laoder(pdf_path : str):

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    print(f'Loaded {len(docs)} documents from PDF')
    for i, doc in enumerate(docs):
        print(f'Document {i+1} Content Preview: {doc.page_content[:100]} ...')
        print(f'Metadata: {doc.metadata}')

if __name__ == "__main__":
    # load_text_file()
    pdf_laoder('docs/langchain_demo.pdf')

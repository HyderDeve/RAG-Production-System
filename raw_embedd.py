from dotenv import load_dotenv
from groq import Groq
from langchain_huggingface import HuggingFaceEmbeddings
import os

load_dotenv()

client1 = Groq(api_key=os.getenv("GROQ_API_KEY"))


client = HuggingFaceEmbeddings(
        model_name = 'Qwen/Qwen3-Embedding-0.6B',
        encode_kwargs = {'normalize_embeddings' : True}    
)

client2 = HuggingFaceEmbeddings(
    model_name = 'BAAI/bge-m3',
    encode_kwargs = {'normalize_embeddings' : True}
)

# conversation = client.chat.completions.create(
#     model="openai/gpt-oss-120b",
#     messages = [
#         {'role' : 'system', 'content':'You are a helpful assistant.'},
#         {'role' : 'user', 'content' : 'What is the capital of France ?'},
#     ],
# )


# print(conversation.choices[0].message.content)

response = client.embed_query(
    text = 'Your string goes here'
)

print(response)
print('\n',len(response))
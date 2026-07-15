from dotenv import load_dotenv
from importlib.metadata import version

load_dotenv()

core_version = version('langchain-core')
lg_version = version('langgraph')

from langchain_groq import ChatGroq



print(f'langchain-core version: {core_version}\n')
print(f'langgraph version: {lg_version}\n')


def main():
    llm = ChatGroq(
        model = 'openai/gpt-oss-120b',
        temperature = 0.5 
    )

    response = llm.invoke('Say "Setup Complete" in One Word')

    print(f'Responce from Groq: {response}')

if __name__ == "__main__":
    main()

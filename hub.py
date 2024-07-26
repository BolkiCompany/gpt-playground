from dotenv import load_dotenv

from langchain import hub


load_dotenv()

print(hub.pull("rlm/rag-prompt"))

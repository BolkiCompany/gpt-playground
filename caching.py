from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache


load_dotenv()

set_llm_cache(InMemoryCache())
model = ChatOpenAI(model="gpt-4o-mini")
prompt = PromptTemplate.from_template("{country} 에 대해서 200자 내외로 요약해줘")

chain = prompt | model

print(chain.invoke({"country": "한국"}))

print(chain.invoke({"country": "한국"}))

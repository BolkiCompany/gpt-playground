from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel


load_dotenv()

model = ChatOpenAI(
    temperature=0.1,
    model="gpt-4o-mini",
)

chain1 = (
    PromptTemplate.from_template("{topic}에 대해 1문장으로 설명해주세요.")
    | model
    | StrOutputParser()
)

chain2 = (
    PromptTemplate.from_template("{topic}에 대해 3문장으로 설명해주세요.")
    | model
    | StrOutputParser()
)

combined_chain = RunnableParallel(c1=chain1, c2=chain2)
print(combined_chain.batch([{"topic": "대한민국"}, {"topic": "미국"}]))

# for r in chain.stream({"topic": "멀티모달"}):
#     print(r, end="", flush=True)

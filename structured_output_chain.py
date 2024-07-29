from dotenv import load_dotenv

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class Quiz(BaseModel):
    question: str = Field(..., description="퀴즈의 질문")
    level: str = Field(..., description="퀴즈의 난이도를 나타냅니다. (쉬움, 보통, 어려움)")
    options: List[str] = Field(..., description="퀴즈의 4개의 선택지입니다.")

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're a world-famous quizzer and generates quizzes in structured formats.",
        ),
        (
            "human",
            "TOPIC 에 제시된 내용과 관련한 4지선다형 퀴즈를 출제해 주세요. 만약, 실제 출제된 기출문제가 있다면 비슷한 문제를 만들어 출제하세요."
            "단, 문제에 TOPIC 에 대한 내용이나 정보는 포함하지 마세요. \nTOPIC:\n{topic}",
        ),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

chain = create_structured_output_runnable(Quiz, llm, prompt)

print(chain.invoke({"topic": "ADSP(데이터 분석 준전문가) 자격 시험"}))
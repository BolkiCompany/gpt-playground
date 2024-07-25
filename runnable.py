from dotenv import load_dotenv
from datetime import datetime

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


model = ChatOpenAI(model="gpt-4o-mini")

# 1. 사실 이렇게하면 그냥 숫자만 넘겨도 됨
chain = (
    {"num": RunnablePassthrough()}
    | PromptTemplate.from_template("{num}의 5승은?")
    | model
)
print(chain.invoke(4).content)

# 2. RunnablePassthrough 예시들
print(RunnablePassthrough().invoke({"num": 4}))
print(RunnablePassthrough.assign(new_num=lambda x: x["num"] * 3).invoke({"num": 1}))

runnable = RunnableParallel(
    passed=RunnablePassthrough(),
    extra=RunnablePassthrough.assign(mul=lambda x: x["num"] * 3),
    modified=lambda x: x["num"] + 1,
)
print(runnable.invoke({"num": 1}))

# 3. RunnableLambda 쓰는 방법


def get_today(temp):
    return datetime.today().strftime("%b-%d")


chain2 = (
    {"today": RunnableLambda(get_today), "n": RunnablePassthrough()}
    | PromptTemplate.from_template("{today}가 생일인 유명인 {n}명 알려줘.")
    | model
    | StrOutputParser()
)
print(chain2.invoke(3))


# 4. itemgetter 사용
from operator import itemgetter


def get_length(text):
    return len(text)


def _multiple_length(text1, text2):
    return len(text1) * len(text2)


def multiple_length(_dict):
    return _multiple_length(_dict["text1"], _dict["text2"])


chain3 = (
    {
        "a": itemgetter("word1") | RunnableLambda(get_length),
        "b": {"text1": itemgetter("word1"), "text2": itemgetter("word2")}
        | RunnableLambda(multiple_length),
    }
    | PromptTemplate.from_template("{a} + {b}는 무엇일까요?")
    | model
)
print(chain3.invoke({"word1": "hello", "word2": "world"}))

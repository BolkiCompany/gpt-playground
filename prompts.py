from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    PromptTemplate,
    load_prompt,
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")
template = "{country}의 수도는?"


# 1. 단순 template
prompt1 = PromptTemplate.from_template(template)
print(prompt1)

prompt2 = prompt1.format(country="미국")
print(prompt2)


# 2. input 유효성 검사
prompt3 = PromptTemplate(template=template, input_variables=["country"])
print(prompt3)


# 3. 반복되는 값은 partial으로 관리
prompt4 = PromptTemplate(
    template="{c1}과 {c2}의 수도는 각각 어디인가요?",
    input_variables=["c1"],
    partial_variables={"c2": "미국"},
)
print(prompt4)
print(prompt4.format(c1="대한민국"))
print(prompt4.partial(c2="캐나다"))  # partial도 수정 가능


# 4. 파일에서 template 가져오기
prompt5 = load_prompt("prompts/fruit.yaml")
print(prompt5)


# 5. ChatPromptTemplate
template = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절한 AI 어시스턴트입니다. 당신의 이름은 {name} 입니다."),
        ("human", "내 이름은 김은종이야."),
        ("ai", "안녕하세요. 무엇을 도와드릴까요?"),
        ("human", "{input}"),
    ]
)
prompt6 = template.format_messages(name="Teddy", input="당신의 나이와 이름을 밝히시오.")
print(model.invoke(prompt6).content)

chain = template | model
print(chain.invoke({"name": "Jimmy", "input": "내 이름은 뭐였지?"}).content)


# 6. MessagePlaceholder - 메시지 삽입
template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 요약 전문 AI 어시스턴트입니다. 당신의 임무는 주요 키워드로 대화를 요약하는 것입니다.",
        ),
        MessagesPlaceholder(variable_name="conversation"),
        ("human", "지금까지 대화를 {count} 단어로 요약해주세요."),
    ]
)
prompt7 = template.format(
    count=5,
    conversation=[
        ("human", "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다."),
        ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
    ],
)

chain = template | model | StrOutputParser()
print(
    chain.invoke(
        {
            "count": 5,
            "conversation": [
                (
                    "human",
                    "안녕하세요! 저는 오늘 새로 입사한 테디 입니다. 만나서 반갑습니다.",
                ),
                ("ai", "반가워요! 앞으로 잘 부탁 드립니다."),
            ],
        }
    )
)

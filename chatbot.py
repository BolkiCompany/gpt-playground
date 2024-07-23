from dotenv import load_dotenv
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    trim_messages,
    AIMessage,
)
from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


load_dotenv()

store = {}
config = {"configurable": {"session_id": "abc2"}}
model = ChatOpenAI(model="gpt-4o-mini")

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages111 = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# message text(tuple) -> template 동적 생성 함수
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        # 이 위치에 messages111 변수로 접근하여 동적 관리
        MessagesPlaceholder(variable_name="messages111"),
    ]
)

chain = (
    RunnablePassthrough.assign(messages111=itemgetter("messages222") | trimmer)
    | prompt
    | model
)

# 1. history 수동 적용
# chain 쓰면 최상단 layer 기준으로 변수명 일치시켜야하는듯..
response = chain.invoke(
    {
        "language": "Korean",
        "messages222": messages111
        + [HumanMessage(content="What math problem did i ask?")],
    }
)
print(response.content)

# 2. History 함수 사용
with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages222",
)

response = with_message_history.invoke(
    {"messages222": [HumanMessage(content="Hi! I'm EJ")], "language": "Korean"},
    config=config,
)
print(response.content)

for r in with_message_history.stream(
    {"messages222": [HumanMessage(content="What's my name?")], "language": "Korean"},
    config=config,
):
    print(r.content, end="|")

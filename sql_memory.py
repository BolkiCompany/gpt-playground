from dotenv import load_dotenv

from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableFieldSpec
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

chat_message_history = SQLChatMessageHistory(
    session_id="sql_history",
    connection="sqlite:///sqlite.db",
)

chat_message_history.add_user_message(
    "안녕? 만나서 반가워. 내 이름은 김은종이야. 나는 랭체인 개발자야. 앞으로 잘 부탁해!"
)

chat_message_history.add_ai_message("안녕 은종, 만나서 반가워. 나도 잘 부탁해!")

prompt = ChatPromptTemplate.from_messages(
    [
        # 시스템 메시지
        ("system", "You are a helpful assistant."),
        # 대화 기록을 위한 Placeholder
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),  # 질문
    ]
)

chain = prompt | ChatOpenAI(model_name="gpt-4o") | StrOutputParser()


def get_chat_history(user_id, conversation_id):
    return SQLChatMessageHistory(
        table_name=user_id,
        session_id=conversation_id,
        connection="sqlite:///sqlite.db",
    )


config_fields = [
    ConfigurableFieldSpec(
        id="user_id",
        annotation=str,
        name="User ID",
        description="Unique identifier for a user.",
        default="",
        is_shared=True,
    ),
    ConfigurableFieldSpec(
        id="conversation_id",
        annotation=str,
        name="Conversation ID",
        description="Unique identifier for a conversation.",
        default="",
        is_shared=True,
    ),
]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="question",
    history_messages_key="chat_history",
    history_factory_config=config_fields,
)

config = {"configurable": {"user_id": "user1", "conversation_id": "conversation1"}}
print(
    chain_with_history.invoke({"question": "안녕 반가워, 내 이름은 김은종이야"}, config)
)
print(chain_with_history.invoke({"question": "내 이름이 뭐라고?"}, config))

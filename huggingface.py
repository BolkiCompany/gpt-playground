import os
from dotenv import load_dotenv
from huggingface_hub import login

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint

load_dotenv()
login(token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

template = """<|system|>
You are a helpful assistant.<|end|>
<|user|>
{question}<|end|>
<|assistant|>"""

prompt = PromptTemplate.from_template(template)

repo_id = "mistralai/Mistral-Nemo-Instruct-2407"

model = HuggingFaceEndpoint(
    repo_id=repo_id,
    max_new_tokens=256,
    temperature=0.1,
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],  # 허깅페이스 토큰
)

chain = prompt | model | StrOutputParser()

print(chain.invoke({"question": "what is the capital of South Korea?"}))

from dotenv import load_dotenv
import base64, httpx

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

image_data = base64.b64encode(
    httpx.get("https://storage.googleapis.com/vectrix-public/fruit/apple.jpeg").content
).decode("utf-8")

message = HumanMessage(
    content=[
        {"type": "text", "text": "describe the fruit in this image"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
        },
    ],
)

# Invoke the model with the message
response = model.invoke([message])

# Print the model's response
print(response.content)

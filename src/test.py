import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
)

LLM_DEPLOYMENT = os.getenv("AZURE_LLM_DEPLOYMENT")

tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"}
            },
            "required": ["city"]
        }
    }
]

response = client.responses.create(
    model=LLM_DEPLOYMENT,
    input="What's the weather in Hanoi?",
    tools=tools,
)

print(json.dumps(response.model_dump(), indent=2, default=str))
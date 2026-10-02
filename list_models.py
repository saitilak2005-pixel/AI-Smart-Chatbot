import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
import os
from openai import OpenAI

key = os.getenv('LLM_API_KEY')
client = OpenAI(api_key=key, base_url='https://api.groq.com/openai/v1')

# List all available models
models = client.models.list()
print("Available Groq models:")
for m in sorted(models.data, key=lambda x: x.id):
    print(f"  {m.id}")

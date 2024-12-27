from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
  organization=os.environ.get("OPEN_API_ORG_ID"),
  project=os.environ.get("OPEN_API_PROJECT_ID"),
  api_key=os.environ.get("OPENAI_API_KEY"),
)

def generate(prompt, dom):
    res = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": f"""Here's the html content: {dom}""",
            }
        ],
        model="gpt-4o",
    )

    dict = res.to_dict()
    return dict['choices'][0]['message']['content'];
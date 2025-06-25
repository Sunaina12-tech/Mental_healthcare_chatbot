
from dotenv import load_dotenv
import os
import streamlit as st
import openai
from utils import preprocess, get_context, decode_response

from openai import OpenAI
client = OpenAI()

response = client.responses.create(
    model="gpt-4.1",
    input="Write a one-sentence bedtime story about a unicorn."
)

print(response.output_text)
# Initialize OpenAI client
# client = openai(api_key="sk-...qqUA")  # Use environment variables in production!
load_dotenv()
openai.api_key=os.getenv('sk-...qqUA')

def generate_response(model: str, input_text: str):
    """Generic function to handle all openai models."""
    if model in ["gpt-3.5-turbo", "gpt-4"]:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": input_text}]
        )
        return response.choices[0].message.content
    elif model in ["dall-e-2", "dall-e-3"]:
        response = client.images.generate(
            model=model,
            prompt=input_text,
            size="1024x1024"
        )
        return response.data[0].url  # Returns image URL
    else:
        raise ValueError(f"Unsupported model: {model}")

def main():
    st.set_page_config(page_title="Mental Health Chatbot")
    st.title("Mental Health Chatbot")

    input_text = st.text_input("You:", key="input_text")

    if input_text:
        input_text = preprocess(input_text)
        context = get_context(input_text)  # Ensure this returns correct model names

        # Map legacy names to current model IDs
        model_mapping = {
            "GPT-3": "gpt-3.5-turbo",
            "DALL-E 2": "dall-e-2",
            "Codex": "gpt-3.5-turbo",  # Codex is deprecated
            "ChatGPT": "gpt-4"
        }

        model = model_mapping.get(context, "gpt-3.5-turbo")
        response = generate_response(model, input_text)
        
        st.text_area("Bot:", value=decode_response(response), height=150, key="bot_response")

if __name__ == "__main__":
    main()

from flask import Flask, render_template, jsonify, request
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

gemini_secret = os.environ["GEMINI_API_KEY"]

# Initializing the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.5,
    max_tokens=500,
    timeout=None,
    max_retries=2,
    api_key=gemini_secret,
)

app = Flask(__name__)


def call_with_template(input_data):
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
            You are an expert blog writer bot. Your task is to generate a well-written blog post based on the topic or explanation provided by the user. 
            The output should be plain text without any HTML, Markdown, or other formatting. 
            Use simple newlines, punctuations, and proper sentence structure to make the text readable. Avoid any bullet points (*), hashes (##), or other formatting styles like bold or italics.
            The content should be well-structured and easy to read, with paragraphs to separate different ideas. Stay within the token limit of 500 tokens.
            """,
        ),
        (
            "human",
            "Write a blog based on the following topic: {input_data}",
        ),
    ])

    # Chain the prompt with the LLM
    chain = prompt | llm

    # Invoke the chain with a specific year
    response = chain.invoke({"input_data": input_data})

    # Print the response
    return response.content


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/generate', methods=['GET', 'POST'])
def generate():
    if request.method == 'POST':
        try:
            prompt = request.json.get('prompt')
            print(f"Prompt: {prompt}")
            response = call_with_template(prompt)
            print(f"Response: {response}")
            return response
        except:
            return jsonify({'error': 'Invalid request'})
    return jsonify({'error': 'Invalid request method'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

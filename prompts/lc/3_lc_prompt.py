from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation"
)

cm = ChatHuggingFace(llm=endpoint)

st.header("AI Literature Assistant")

topic = st.text_input("Enter topic")
number_of_lines = st.text_input("Enter number of lines")
style = st.text_input("Enter style")
language = st.text_input("Enter language")

template = load_prompt("prompt_1.json")

# 🔥 Create chain
chain = template | cm

if st.button("Generate"):
    result = chain.invoke({
        "style": style,
        "number_of_lines": number_of_lines,
        "topic": topic,
        "language": language
    })

    st.write(result.content)
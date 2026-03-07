from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint( repo_id="Qwen/Qwen2.5-1.5B-Instruct",
task="text-generation", )

cm = ChatHuggingFace(llm=endpoint)
st.header("AI Litrature Assistant")
topic = st.text_input("Enter topic")

number_of_line = st.text_input("Enter number of lines: ")
Style = st.text_input("Enter style: ")
language = st.text_input("Enter language: ")

system_prompt = f"you are a helpful hystery assistant. you need to write on the topic - {topic} in {Style} style in {number_of_line} lines"
if st.button("Generate"): 
    result = cm.invoke(system_prompt)
    st.write(result.content)
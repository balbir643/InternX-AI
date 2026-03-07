from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint( repo_id="Qwen/Qwen2.5-1.5B-Instruct",
task="text-generation", )
cm = ChatHuggingFace(llm=endpoint)

template = PromptTemplate( 
    template=""" 
    You are a highly skilled literature assistant.
 TASK: 
 - Answer the user's question in the style of: {style} 
 - The response must be exactly {number_of_lines} lines long.
 - Every line should be meaningful and revelant.
 - The content must be focused only on th topic: {topic} 
 - Do not include anything outside the topic.
 LANGUAGE RULE: 
 - Response in only {language} language.
 - Do not use any language other than {language} unless they are unavoidable proper nouns.
 FORMAT RULES: 
 - Do not include heading, explanations, or extra notes. - Always start new sentences from a new line.
  QUALITY RULE: 
  - Ensure the respone matches the tone, and vocabulary. 
  - Avoid repetition and keep the writing neutral.
 Before producing the final answer, internally verify:
  1. Style is correct.
  2. Topic is respected.
  3. Line count is exact.
  4. Language is correct. 
 Now generate the final response.
  """, 
 input_variables=["style", "number_of_lines", "topic", "language"],
)

st.header("AI Litrature Assistant")
topic = st.text_input("Enter topic")
number_of_lines = st.text_input("Enter number of lines")
style = st.text_input("Enter style")
language = st.text_input("Enter langauge")

system_prompt = template.invoke({
    "style": style,
    "number_of_lines": number_of_lines,
    "topic": topic,
    "language": language
})
if st.button("generate"):
    result = cm.invoke(system_prompt)
    (result.content)
    st.write(result.content)

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import load_prompt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-1.5B-Instruct", task="text-generation")
cm = ChatHuggingFace(llm=endpoint)

st.header("AI Advanced Energy Saving and Smart Power Management Assistant")


number_of_daily_unit_consumption = st.text_input("Enter number of daily unit consumption")
electricity_price_per_unit = st.text_input("Enter electricity price per unit")
emission_factor = st.text_input("Enter emission factor")
number_of_ac_rooms = st.text_input("Enter number of ac rooms")
number_of_non_ac_rooms = st.text_input("Enter number of non ac rooms")
language = st.text_input("Enter language")


template = load_prompt("prompt_8.json")
chain = template | cm
if st.button("generate"):
    result = chain.invoke({
             "number_of_daily_unit_consumption": number_of_daily_unit_consumption,
            "electricity_price_per_unit": electricity_price_per_unit,
            "emission_factor": emission_factor,
            "number_of_ac_rooms": number_of_ac_rooms,
            "number_of_non_ac_rooms": number_of_non_ac_rooms,
            "language": language
    })
    st.write(result.content)

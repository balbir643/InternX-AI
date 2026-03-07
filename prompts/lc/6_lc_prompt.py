import json
import streamlit as st
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

st.set_page_config(page_title="AI LinkedIn Post Generator", page_icon="🚀")

st.title("🚀 AI LinkedIn Post Generator")

# Load JSON prompt
with open("prompt_12.json", "r") as file:
    data = json.load(file)

template = data["template"]

# Prompt Template
prompt = PromptTemplate(
    input_variables=[
        "topic",
        "experience_level",
        "tone",
        "emoji_option",
        "hook_optimizer",
        "engagement_booster",
        "viral_toggle"
    ],
    template=template
)

# Load HuggingFace model
endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=endpoint)

# Output parser
parser = StrOutputParser()

# LangChain chain
chain = prompt | model | parser


# ---------- Streamlit UI ----------

st.sidebar.header("🎨 Customize Output Colors")

text_color = st.sidebar.color_picker("Pick Text Color", "#0e1117")
bg_color = st.sidebar.color_picker("Pick Background Color", "#f0f2f6")

topic = st.text_input("Topic")

experience_level = st.selectbox(
    "Experience Level",
    ["Student", "Beginner", "Professional", "Expert", "Leader"]
)

tone = st.selectbox(
    "Tone",
    ["Professional", "Inspirational", "Educational", "Storytelling", "Thought Leadership"]
)

emoji_option = st.selectbox(
    "Add Emojis",
    ["Yes", "No"]
)

hook_optimizer = st.selectbox(
    "Hook Optimizer",
    ["ON", "OFF"]
)

engagement_booster = st.selectbox(
    "Engagement Booster",
    ["ON", "OFF"]
)

viral_toggle = st.selectbox(
    "Make It Viral",
    ["ON", "OFF"]
)

# Generate button
if st.button("Generate Post"):

    if topic.strip() == "":
        st.warning("⚠️ Please enter a topic.")
    else:
        with st.spinner("Generating your LinkedIn post..."):
            result = chain.invoke({
                "topic": topic,
                "experience_level": experience_level,
                "tone": tone,
                "emoji_option": emoji_option,
                "hook_optimizer": hook_optimizer,
                "engagement_booster": engagement_booster,
                "viral_toggle": viral_toggle
            })

        st.subheader("✨ Generated LinkedIn Post")

        # Styled Output Box
        st.markdown(
            f"""
            <div style="
                background-color:{bg_color};
                padding:25px;
                border-radius:15px;
                color:{text_color};
                font-size:17px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                white-space: pre-wrap;
            ">
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )
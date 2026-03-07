from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint 
from dotenv import load_dotenv

load_dotenv()

endpoint = HuggingFaceEndpoint( repo_id="Qwen/Qwen2.5-1.5B-Instruct",
task="text-generation", )

cm = ChatHuggingFace(llm=endpoint)
topic = input("Enter topic: ")

number_of_line = input("Enter number of lines: ")
Style = input("Enter style: ")
language = input("Enter language: ")
print()
system_prompt = f"you are a helpful hystery assistant. you need to write on the topic - {topic} in {Style} style in {number_of_line} lines"
result = cm.invoke(system_prompt)
print(result.content)
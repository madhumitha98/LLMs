import os
import dotenv
import openai
dotenv.load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain 
import streamlit as st
from langchain.prompts import PromptTemplate 

openai.api_key = os.getenv('OPENAI_API_KEY')

st.title('Celebrity Search')

input_text = st.text_input("What to know about a celebrity")

gpt = ChatOpenAI()

prompt = (
    PromptTemplate.from_template("Who is {topic}" + ", make it short" )
)

prompt.format(topic = input_text)

chain = LLMChain(llm = gpt, prompt = prompt)
response = chain.run(topic = input_text  )
st.write(response)
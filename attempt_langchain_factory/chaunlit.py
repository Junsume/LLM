import os
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import LLMChain
from getpass import getpass

import chainlit as cl

# Prompt the user to enter the Hugging Face API token
HUGGINGFACEHUB_API_TOKEN = getpass("Enter your Hugging Face API token: ")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

template = """Question: {question}

Answer: Let's think step by step ."""


@cl.lanchain_factory(use_async=True)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=HuggingFaceHub(temperature=0, verbose=True))

    return llm_chain

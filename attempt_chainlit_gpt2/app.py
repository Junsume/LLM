import os
import chainlit as cl
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

# from getpass import getpass
from dotenv import load_dotenv

# Prompt the user to enter the Hugging Face API token
# HUGGINGFACEHUB_API_TOKEN = getpass("Enter your Hugging Face API token: ")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

model_id = "openai-community/gpt2-medium"
conv_model = HuggingFaceHub(
    huggingfacehub_api_token=HUGGINGFACE_API_TOKEN,
    repo_id=model_id,
    model_kwargs={"temperature": 0.6, "max_new_tokens": 250},
)

template = """You are a helpful AI assistant that completes a story based on the query received as input.

{query}
"""


@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=template, input_variables=["query"])
    llm_chain = LLMChain(llm=conv_model, prompt=prompt, verbose=True)
    cl.user_session.set("llm_chain", llm_chain)


# @cl.on_message
# async def main(message: str):
#     # Retrieve the chain from the user session
#     llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain

#     # Call the chain asynchronously
#     res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

#     # Do any post processing here

#     # Send the response
#     await cl.Message(content=res["text"]).send()


@cl.on_message
async def handle_message(message: cl.Message):
    llm_chain = cl.user_session.get("llm_chain")
    res = await llm_chain.acall(
        inputs={"query": message.content},
        callbacks=[cl.AsyncLangchainCallbackHandler()],
    )
    response_text = (
        res["text"].split("\n", 1)[-1].strip()
    )  # Extract only the response text after the first line
    await cl.Message(content=response_text).send()

import os
from typing import List
import chainlit as cl
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain.docstore.document import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from getpass import getpass

# Prompt the user to enter the Hugging Face API token
HUGGINGFACEHUB_API_TOKEN = getpass("Enter your Hugging Face API token: ")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
huggingfacehub_api_token = (os.environ["HUGGINGFACEHUB_API_TOKEN"],)

model_id = "gpt2-medium"
conv_model = HuggingFaceHub(
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    repo_id=model_id,
    model_kwargs={"temperature": 0.6, "max_new_tokens": 250},
)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)


@cl.on_chat_start
async def on_chat_start():
    res = await cl.AskUserMessage(
        content="What is your name?", author="Jun's bot", timeout=30
    ).send()
    if res:
        await cl.Message(
            content=f"hello {res['output']}!,{res}", author="Jun's bot"
        ).send()

    # Ask for a file
    files = None

    # wait for the user to uploade a file
    while files == None:
        files = await cl.AskFileMessage(
            content="Please uploade a text file to begin.",
            accept=["text/plain", ".csv"],
            max_size_mb=20,
            timeout=180,
        ).send()

    # decode the file
    file = files[0]
    msg = cl.Message(content=f"Processing `{file.name}`...", disable_feedback=True)
    await msg.send()
    with open(file.path, "r", encoding="utf-8") as f:
        text = f.read()

    # await cl.Message(
    #     content=f"'{file.name}' uploaded, it contains {len(text)} characters!"
    # ).send()

    # Split the text into chunks
    texts = text_splitter.split_text(text)

    # Create a metadata for each chunk
    metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

    # Create a Chroma vector store
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_texts)(
        texts, embeddings, metadatas=metadatas
    )

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        conv_model,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: ConversationalRetrievalChain
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(
                    content=source_doc.page_content, name=source_name, display="side"
                )
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()

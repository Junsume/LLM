import chainlit as cl


# 1
@cl.on_message
# The main function will be called every time a user inputs a message in the chatbot UI
async def main(message: cl.Message):
    # The Message class is responsible for sending a reply back to the user
    # You can put your custom logic within the function to process the user’s input,
    # such as analyzing the text, calling an API, or computing a result.

    # In this example, we simply send a message containing the user’s input.
    await cl.Message(content=f"Received: {message.content}").send()


## 2
## asking name
# @cl.on_chat_start
# async def main():
#     res = await cl.AskUserMessage(content="What is your name?", author="Jun's bot", timeout=30).send()
#     if res:
#         await cl.Message(content=f"hello {res['output']}!").send()


# TO RUN
# chainlit run app.py -w (The -w flag tells Chainlit to enable auto-reloading)

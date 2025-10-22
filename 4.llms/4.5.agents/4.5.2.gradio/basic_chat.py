import gradio as gr
from dotenv import load_dotenv
import random
import dspy

load_dotenv()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

qa = dspy.Predict("chat_history, question -> answer")

chat_history = dspy.History(messages=[])

def random_reponse(message, history):
    outputs = qa(chat_history=chat_history, question=message)
    chat_history.messages.append({"question": message, **outputs})
    return outputs.answer

demo = gr.ChatInterface(random_reponse, type="messages")

demo.launch()
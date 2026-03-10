import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot")
st.write("Chat with the AI below")

# Load model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="microsoft/DialoGPT-medium")

chatbot = load_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
prompt = st.chat_input("Type your message...")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    response = chatbot(prompt, max_length=100, num_return_sequences=1)

    bot_reply = response[0]["generated_text"]

    # Display bot reply
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

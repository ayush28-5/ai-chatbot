import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Chatbot", page_icon="🤖")

st.title("🤖 AI Chatbot")
st.write("Chat with the AI below")

# Load AI model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="microsoft/DialoGPT-medium")

chatbot = load_model()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Reset chat button
if st.button("🔄 Reset Chat"):
    st.session_state.messages = []
    st.rerun()

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input box
prompt = st.chat_input("Type your message...")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Build conversation context
    conversation = ""
    for msg in st.session_state.messages:
        conversation += msg["content"] + " "

    # Generate response
    response = chatbot(conversation, max_length=120, num_return_sequences=1)

    bot_reply = response[0]["generated_text"].replace(conversation, "").strip()

    # Display bot reply
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

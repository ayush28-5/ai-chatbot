import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="wide")

st.title("🤖 AI Chatbot")

# Load model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="microsoft/DialoGPT-medium")

chatbot = load_model()

# Initialize session storage
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"


# Sidebar
st.sidebar.title("💬 Chats")

chat_names = list(st.session_state.chats.keys())

selected_chat = st.sidebar.selectbox(
    "Select Chat",
    chat_names,
    index=chat_names.index(st.session_state.current_chat)
)

st.session_state.current_chat = selected_chat

# New chat button
if st.sidebar.button("➕ New Chat"):
    new_chat_name = f"Chat {len(chat_names) + 1}"
    st.session_state.chats[new_chat_name] = []
    st.session_state.current_chat = new_chat_name
    st.rerun()

messages = st.session_state.chats[st.session_state.current_chat]

# Display messages
for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Type your message...")

if prompt:
    messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    conversation = ""
    for msg in messages:
        conversation += msg["content"] + " "

    response = chatbot(conversation, max_length=120, num_return_sequences=1)

    bot_reply = response[0]["generated_text"].replace(conversation, "").strip()

    messages.append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.markdown(bot_reply)

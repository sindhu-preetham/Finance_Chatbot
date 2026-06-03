import streamlit as st

st.title("Finance Chatbot")

url1 = st.sidebar.text_input("URL 1")

question = st.text_input(
    "Ask a question"
)

if question:
    st.write("You asked:", question)
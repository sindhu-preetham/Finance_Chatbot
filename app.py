import streamlit as st
from modules.retrieval import get_retrieval_response

st.set_page_config(
    page_title="Finance News Chatbot",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Finance News Chatbot")
st.markdown(
    "Ask questions about financial news articles."
)

# Sidebar
st.sidebar.header("Article URLs")

urls = []

for i in range(1, 4):
    url = st.sidebar.text_input(
        f"Article URL {i}"
    )

    if url:
        urls.append(url)

process_btn = st.sidebar.button(
    "Process Articles"
)

clear_btn = st.sidebar.button(
    "Clear Database"
)

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Process URLs
if process_btn:

    if len(urls) == 0:
        st.warning("Please enter at least one URL")

    else:
        with st.spinner(
            "Downloading and processing articles..."
        ):

            # pipeline call here
            # load_data(urls)
            # chunk_data()
            # create_embeddings()
            # save_to_faiss()

            st.success(
                f"{len(urls)} articles processed successfully"
            )

# Question section
question = st.text_input(
    "Ask a question about the articles"
)

if st.button("Submit Question"):

    if question:

        with st.spinner(
            "Generating answer..."
        ):

            # answer = get_answer(question)

            response = get_retrieval_response(
                question
            )

            if response["success"]:

                answer = response["context"]

            else:

                answer = (
                    "No relevant information found."
                )

            st.session_state.messages.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

# Chat history
for chat in st.session_state.messages:

    st.markdown(
        f"**Question:** {chat['question']}"
    )

    st.markdown(
        f"**Answer:** {chat['answer']}"
    )

    st.divider()
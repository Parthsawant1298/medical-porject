import streamlit as st
from rag import user_input

def main():
    st.set_page_config(page_title="Chat BOT")
    st.header("Chat with chat bot💁")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_question = st.text_input("Ask a Question related to medical")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        response = user_input(user_question)
        st.session_state.messages.append({"role": "bot", "content": response})

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div style='text-align: right;'><b>User:</b> {message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left;'><b>Bot:</b> {message['content']}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

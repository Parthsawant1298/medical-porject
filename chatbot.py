from rag import user_input
import streamlit as st

def main():
    st.set_page_config(page_title="Medico")
    st.title("💬 Chatbot")
    st.caption("A Medical chatbot powered by Gemini")

    
    # Chatbot interface using streamlit in the main function
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I am a medical chatbot. How can I help you today?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_question := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": user_question})
        st.chat_message("user").write(user_question)
        response = user_input(user_question)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    
if __name__ == "__main__":
    main()










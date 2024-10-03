import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Directly set the API key
api_key = "AIzaSyDoR10wPWSnCCLXHZWWrlrAg7XCXFzzpx8"  # Replace with your actual Google API key

# Configure the API key
genai.configure(api_key=api_key)

# Initialize Streamlit page configuration
st.set_page_config(page_title="Medico")
st.title("🩺 Chatbot")
st.caption("Medical chatbot powered by Gemini✨")

# Initialize session state for storing messages and vector store
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am a medical chatbot. How can I help you today?"}]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Function to load FAISS vector store
def load_faiss_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    save_path = os.path.expanduser("~/faiss_index")  # Change to your writable path
    if os.path.exists(save_path):
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    else:
        return None

# Load the FAISS vector store if it exists
st.session_state.vector_store = load_faiss_vector_store()

# Function to create and save FAISS vector store
def create_faiss_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    save_path = os.path.expanduser("~/faiss_index")  # Change to a writable path
    vector_store.save_local(save_path)
    st.session_state.vector_store = vector_store  # Store in session state for later use

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to handle user input and generate responses
def user_input(user_question):
    if st.session_state.vector_store is None:
        st.error("Vector store is not available. Please create it first.")
        return "Error: Vector store not initialized."

    docs = st.session_state.vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to load conversational AI chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, making sure to provide all the details. If the answer is not in
    the provided context, just say, "answer is not available in the context"; don't provide a wrong answer.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Display all messages in the chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle new user input
if user_question := st.chat_input("Ask a question about medical topics..."):
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_question})
    st.chat_message("user").write(user_question)

    # Generate the response based on the user's question
    response = user_input(user_question)
    
    # Add assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)

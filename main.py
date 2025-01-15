import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Configure Streamlit page
st.set_page_config(page_title="Website Chatbot", page_icon="🤖")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def load_and_process_website(url):
    """Load and process website content"""
    loader = WebBaseLoader(url)
    data = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(data)
    
    return splits

def initialize_chatbot(splits):
    """Initialize the chatbot with website content"""
    # Set Google API key
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    # Initialize LLM and memory
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-002")
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Create conversational chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    
    return qa

# Streamlit UI
st.title("💬 Website Chatbot")

# URL input
url = st.text_input("Enter website URL:", "https://www.buildfastwithai.com/")

if url:
    if "qa_chain" not in st.session_state:
        with st.spinner("Processing website content..."):
            splits = load_and_process_website(url)
            st.session_state.qa_chain = initialize_chatbot(splits)
        st.success("Website processed! You can now ask questions.")

    # Chat interface
    if prompt := st.chat_input("Ask a question about the website"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain({"question": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use:
1. Enter a website URL
2. Wait for the content to be processed
3. Ask questions about the website content
""")

import streamlit as st
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
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
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(data)
        
        return splits
    except Exception as e:
        st.error(f"Error loading website: {str(e)}")
        return None

def initialize_chatbot(splits):
    """Initialize the chatbot with website content"""
    try:
        # Set Google API key
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
        
        # Create embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(splits, embeddings)
        
        # Initialize LLM and memory
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0.7,
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
                "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            }
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create conversational chain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            memory=memory,
            verbose=True
        )
        
        return qa
    except Exception as e:
        st.error(f"Error initializing chatbot: {str(e)}")
        return None

# Streamlit UI
st.title("💬 Website Chatbot")

# Add sidebar with info
with st.sidebar:
    st.markdown("""
    ### About
    This chatbot can answer questions about any website you provide.
    It uses:
    - Gemini Pro for chat
    - FAISS for vector search
    - LangChain for the framework
    """)

# URL input section
url = st.text_input("Enter website URL:", placeholder="Enter URL and click Load Website")
load_button = st.button("Load Website")

if load_button and url:
    # Reset session state if a new website is loaded
    if "qa_chain" in st.session_state:
        del st.session_state.qa_chain
        st.session_state.messages = []
    
    with st.spinner("Processing website content..."):
        splits = load_and_process_website(url)
        if splits:
            qa_chain = initialize_chatbot(splits)
            if qa_chain:
                st.session_state.qa_chain = qa_chain
                st.success("Website processed! You can now ask questions.")
            else:
                st.error("Failed to initialize chatbot. Please check your API key and try again.")
elif load_button and not url:
    st.warning("Please enter a URL first!")

# Chat interface
if "qa_chain" in st.session_state:
    if prompt := st.chat_input("Ask a question about the website"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.qa_chain({"question": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Add footer with instructions
st.markdown("---")
st.markdown("""
### How to use:
1. Enter a website URL in the input field
2. Click the 'Load Website' button
3. Wait for the content to be processed
4. Ask questions about the website content

Note: Make sure you have set up your Google API key in Streamlit secrets.
""")

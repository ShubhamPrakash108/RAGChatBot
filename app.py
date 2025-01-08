import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize session state for conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Function to initialize vector embeddings
def setup_vector_embeddings():
    if "vector_store" not in st.session_state:
        embedding_model = OllamaEmbeddings(model="all-minilm")
        pdf_loader = PyPDFDirectoryLoader("pdfs")
        documents = pdf_loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)
        split_documents = text_splitter.split_documents(documents[:50])
        faiss_store = FAISS.from_documents(split_documents, embedding_model)
        st.session_state.vector_store = faiss_store

# Title and description of the app
st.title("RAG Document Q&A With Groq And Ollama")

# Initialize LLM model
chat_llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.2-1b-preview")

# Create prompt template
chat_prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant designed to answer questions based on the provided context. 
    Do not provide any additional information or assumptions outside the given context.

    <Context>
    {context}
    </Context>

    Question: {input}

    Provide a concise answer:
""")

# User input for querying
user_query = st.text_input("Ask from LLM: ")

# Button for creating the vector embeddings
if st.button("Create Document Embedding"):
    setup_vector_embeddings()
    st.write("VectorDB is ready😍😍😍!!!, Now you can ask questions.")

# If the user has entered a question
if user_query:
    # Create document chain for Q&A
    document_chain = create_stuff_documents_chain(chat_llm, chat_prompt)
    retriever = st.session_state.vector_store.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response_data = retrieval_chain.invoke({'input': user_query})

    # Append the interaction to history
    st.session_state.conversation_history.append({"user": user_query, "response": response_data['answer']})

    # Display the answer
    st.write(response_data['answer'])

    # Show conversation history
    with st.expander("Conversation History"):
        for entry in st.session_state.conversation_history:
            st.write(f"**User:** {entry['user']}")
            st.write(f"**LLM:** {entry['response']}")

    # Format the conversation history for download
    history_text = "\n".join([f"User: {entry['user']}\nLLM: {entry['response']}\n" for entry in st.session_state.conversation_history])

    # Button to download conversation history
    st.download_button(
        label="Download Conversation History",
        data=history_text,
        file_name="conversation_history.txt",
        mime="text/plain"
    )

    # Display documents related to the response
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response_data['context']):
            st.write(doc.page_content)

import streamlit as st
import os
from PIL import Image
from pathlib import Path
# from dotenv import load_dotenv # Uncomment if using .env file
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma # Remove Chroma import
from langchain_community.vectorstores import FAISS # Import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile # To handle uploaded file temporarily

from streamlit import columns

# --- Configuration ---
# If using .env file, uncomment this line
# load_dotenv()

# Configure your Google API Key
# It's recommended to set this as a Streamlit secret named GOOGLE_API_KEY
# Accessing secrets in Streamlit: https://docs.streamlit.io/develop/concepts/connections/secrets
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY # Set as environment variable for Langchain
except KeyError:
    st.error("Error: GOOGLE_API_KEY not found in Streamlit secrets.")
    st.info("Please add your Google API Key to Streamlit secrets.toml file.")
    st.stop() # Stop the app if API key is not set

# --- Streamlit App Title and File Uploader ---
# Page config and styling (optional)
st.set_page_config(page_title="💖 Resume Analyzer", page_icon="🌷")

# Load image (optional)
current_dir = Path(__file__).parent
image_path = current_dir / "assets" / "employer.jpg"
col1,col2 = columns(2)
with col1:
    if image_path.exists():
        img = Image.open(image_path)
        st.image(img, width=200, caption="Your AI friend!")
    else:
        st.subheader("Your AI friend!")
with col2:
     st.title("✨ Welcome to the Resume Analyzer ✨")
     st.subheader("Let's find out if this person is the best fit for the job 💁‍♀️")

uploaded_file = st.file_uploader("Upload your PDF Resume", type="pdf")

# --- Process Uploaded File ---
if uploaded_file is not None:
    # To use PyPDFLoader, we need a file path. We'll save the uploaded file temporarily.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    st.success(f"Successfully uploaded {uploaded_file.name}")

    # --- PDF Loading and Processing ---
    try:
        st.write("Loading document...")
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        st.write(f"Document loaded successfully. Found {len(documents)} pages.")
    except Exception as e:
        st.error(f"Error loading PDF file: {e}")
        os.remove(tmp_file_path) # Clean up temporary file
        st.stop()


    # Split the document into chunks
    st.write("Splitting document into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    document_chunks = text_splitter.split_documents(documents)
    st.write(f"Document split into {len(document_chunks)} chunks.")

    # --- Generate and Store Embeddings (using Langchain) ---
    # Cache this step so it doesn't rerun on every interaction
    @st.cache_resource
    def get_vectorstore(_chunks): # Added underscore here
        print("Loading Langchain embedding model...")
        embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        print("Embedding model loaded.")

        print(f"Creating FAISS vector store and adding {len(_chunks)} document chunks...") # Updated print
        # Use FAISS.from_documents instead of Chroma.from_documents
        vectorstore = FAISS.from_documents(
            documents=_chunks,
            embedding=embedding_model
        )
        print("Document chunks added to FAISS vector store.") # Updated print
        return vectorstore

    vectorstore = get_vectorstore(document_chunks)
    st.write("Embeddings generated and stored.")

    # --- Build Retrieval System ---
    retriever = vectorstore.as_retriever()
    st.write("Retriever created.")

    # --- Integrate with LLM (using Langchain) ---
    # Cache this step as well
    @st.cache_resource
    def get_rag_chain(_retriever): # Added underscore here
        print("Setting up Language Model and RAG chain...")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

        prompt = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context:
        {context}

        Question: {question}
        """)

        rag_chain = (
            {"context": _retriever, "question": RunnablePassthrough()} # Use the underscored variable here too
            | prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain created.")
        return rag_chain

    rag_chain = get_rag_chain(retriever)
    st.write("RAG chain ready.")


    # --- Streamlit Chat Interface ---
    st.subheader("Ask a question about the resume:")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What can I help you with?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                 try:
                    response = rag_chain.invoke(prompt)
                    st.markdown(response)
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                 except Exception as e:
                    st.error(f"An error occurred while generating the response: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})


    # Clean up the temporary file after the script finishes (Streamlit reruns)
    # This might need more robust handling in production, but works for a simple case.
    # A better approach for persistent storage in Streamlit is often needed for larger apps.
    if os.path.exists(tmp_file_path):
       # This block might not always execute reliably on rerun, consider session state for file path
       pass # We'll rely on tempfile's delete=False for now, manual cleanup might be needed.
       # To ensure cleanup, you might need to store the tmp_file_path in session_state
       # and clean up when a new file is uploaded or session ends.
       # For this example, let's keep it simple. The temp files will likely persist
       # until the Streamlit server is stopped.

else:
    st.info("Please upload a PDF file to start.")

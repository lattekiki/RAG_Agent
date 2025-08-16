# RAG_Agent
# 💖 Resume Analyzer

This Streamlit app allows users to upload a PDF resume, analyzes the content using state-of-the-art NLP models, and lets users ask questions about the resume content. It integrates PDF loading, text chunking, embeddings, and a retrieval-augmented generation (RAG) chain with a Google generative language model.

---

## Features

- Upload and process PDF resumes.
- Split large documents into manageable chunks.
- Generate semantic embeddings for document chunks using SentenceTransformer.
- Use FAISS vector store for efficient similarity search.
- Integrate Google Gemini LLM (via Langchain) for question answering.
- Interactive chat interface for asking questions about the resume.

---

## Tools & Libraries Used

| Tool/Library                      | Purpose                                                      |
|----------------------------------|--------------------------------------------------------------|
| [Streamlit](https://streamlit.io)           | Web app framework for building interactive UI               |
| [PIL (Pillow)](https://python-pillow.org)        | Image loading and display                                     |
| [Langchain](https://python.langchain.com)          | Framework for building language model pipelines              |
| - `PyPDFLoader` (langchain_community)     | Load and parse PDF documents                                  |
| - `RecursiveCharacterTextSplitter`        | Split long texts into chunks with overlap                    |
| - `SentenceTransformerEmbeddings`          | Generate semantic embeddings using pre-trained transformer   |
| - `FAISS` (langchain_community)             | Vector store for efficient similarity search                 |
| - `ChatGoogleGenerativeAI`                   | Google Gemini LLM integration for generative responses       |
| - `ChatPromptTemplate` & `RunnablePassthrough`     | Prompt templating and chaining LLMS                           |
| Python `tempfile`                              | Handle temporary file storage for uploaded PDFs              |
| `os` & `pathlib`                               | File and environment management                               |

---

## How It Works - Logic Overview

1. **Google API Key Setup**  
   The app retrieves the Google API key from Streamlit's secret management and sets it as an environment variable for Langchain usage. The app stops if the key is missing.

2. **User Interface**  
   Displays a welcome message with an optional image and allows the user to upload a PDF resume file.

3. **File Handling**  
   The uploaded PDF is temporarily saved to disk to be compatible with Langchain's PDF loader.

4. **PDF Loading and Splitting**  
   - The PDF is loaded into memory using `PyPDFLoader`.
   - The text is split into overlapping chunks (~1000 characters with 200 char overlap) using `RecursiveCharacterTextSplitter` to maintain context for embedding.

5. **Embedding Generation**  
   - Uses `SentenceTransformerEmbeddings` with the `"all-MiniLM-L6-v2"` model to generate vector embeddings of each chunk.
   - Embeddings are stored in a FAISS vector store for fast retrieval based on similarity.

6. **Retriever Setup**  
   - A retriever interface is created from the FAISS vector store to fetch relevant chunks for given queries.

7. **RAG Chain Construction**  
   - A retrieval-augmented generation (RAG) chain is built using Langchain primitives:
     - The retriever provides context.
     - A prompt template instructs the model to answer questions based solely on the retrieved context.
     - The Google Gemini model (`ChatGoogleGenerativeAI`) generates the answer.
     - Output is parsed as a string for display.

8. **Interactive Chat Interface**  
   - Users ask questions about the resume via a chat input.
   - The RAG chain is invoked to generate answers.
   - Chat history is maintained and displayed for conversational context.

9. **Temporary File Cleanup**  
   - The app currently relies on temporary files persisting for the session.
   - Notes are included for potential improvements in file management.

---

## How to Run Locally

1. Clone the repo and navigate to the project folder.

2. Install dependencies (using pip or your preferred package manager):

   ```bash
   pip install streamlit pillow langchain langchain-community langchain-google-genai faiss-cpu sentence-transformers

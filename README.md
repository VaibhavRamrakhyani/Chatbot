# Chatbot with Streamlit

Welcome to the PDF Chatbot project! This application allows users to interact with a conversational AI that can answer questions based on the content of multiple PDF documents. The chatbot is built using Streamlit and integrates various LangChain components, Hugging Face models, and PDF processing libraries.

<img width="960" alt="image" src="https://github.com/user-attachments/assets/b8e1d742-d6cb-4da1-ac0f-9c224e995b74">

## Features

- **PDF Upload:** Upload and process multiple PDF documents.
- **Text Extraction:** Extracts text from PDFs and splits it into manageable chunks.
- **Embeddings and Vectorstore:** Uses Hugging Face embeddings and FAISS for text vectorization and retrieval.
- **Conversational AI:** Employs a conversational AI model to provide responses based on the uploaded PDFs.

## Installation

Ensure you have Python 3.9 installed. Create a virtual environment and install the required packages using the following commands:

```bash
pip install -r requirements.txt
requirements.txt

Copy code
streamlit
python-dotenv
PyPDF2
langchain
faiss-cpu
huggingface-hub
Setup
Hugging Face API Token: Create a .env file in the root directory of the project and add your Hugging Face API token:

dotenv
Copy code
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
Run the Application: Start the Streamlit app with the following command:

bash
Copy code
streamlit run app.py
How It Works
Upload PDFs: Use the sidebar to upload one or more PDF files and click "Process".
Process PDFs: The application extracts and processes text from the uploaded PDFs, splits it into chunks, and creates a vector store for efficient search and retrieval.
Ask Questions: Enter your questions in the text input field to interact with the chatbot. The chatbot will search for relevant information from the PDFs and provide a response.
File Descriptions
app.py: The main script that defines the Streamlit app, handles PDF uploads, processes text, and manages the conversation with the chatbot.
requirements.txt: Lists the required Python packages for the project.      

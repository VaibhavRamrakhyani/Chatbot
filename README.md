# Chatbot with Streamlit

Welcome to the PDF Chatbot project! This application allows users to interact with a conversational AI that can answer questions based on the content of multiple PDF documents. The chatbot is built using Streamlit and integrates various LangChain components, Hugging Face models, and PDF processing libraries.

<img width="960" alt="image" src="https://github.com/user-attachments/assets/b8e1d742-d6cb-4da1-ac0f-9c224e995b74">


# Chatbot Streamlit Application
https://www.loom.com/share/2f8a3f04763347ba9c3200ab9d675fb3?sid=590c5096-84c2-4a26-abf2-40e20e411edc
## Overview

This Streamlit application is a chatbot that can respond to user queries based on the content of an uploaded PDF file. It utilizes the `ollama` library for generating responses and `PyPDF2` for PDF text extraction.

## Prerequisites

Ensure you have the following installed:

- Python 3.7 or higher
- Streamlit
- Ollama
- PyPDF2

## Installation

### 1. Clone the Repository (if applicable)

If you have a Git repository, clone it using:
```bash
git clone <repository-url>
cd <repository-directory>
2. Create and Activate a Virtual Environment
Create a virtual environment (optional but recommended):


Copy code
python -m venv venv
Activate the virtual environment:

On Windows:

Copy code
venv\Scripts\activate
On macOS/Linux:
Copy code
source venv/bin/activate


3. Install Dependencies
Install the required Python libraries:

Copy code
pip install streamlit ollama PyPDF2

4.Setting Up Ollama and Downloading the Model
1. Install Ollama CLI
Download and install the Ollama CLI by following the instructions on the Ollama website ["https://ollama.com/download"]. The installation process may vary based on your operating system:

For Windows:

Download the installer from the Ollama website and follow the installation instructions.
For macOS/Linux:

Use Homebrew or download the binary from the Ollama website.
2. Download the gemma:2b Model
Once you have Ollama installed, use the following command to download the gemma:2b model:

bash
Copy code
ollama model download gemma:2b
Running the Application
Save Your Code

Save the provided code in a file, e.g., chatbot.py.

Start the Ollama server:
in cmd promt:
ollam serve

Start the Streamlit Application

Run the Streamlit application with:
Copy code
python -m streamlit run chatbot.py

Access the Application

Open your web browser and navigate to the URL provided in the terminal, usually http://localhost:8501.

How to Use the Application
Upload a PDF

Use the "Upload a PDF file" button to upload a PDF. The application will process the PDF and extract text from it.

Interact with the Chatbot

Type your queries into the chat input box and press Enter. The chatbot will respond based on the content of the uploaded PDF.

Handling Responses

If the chatbot cannot find relevant information in the PDF, it will prompt you to contact the administrator or the company.

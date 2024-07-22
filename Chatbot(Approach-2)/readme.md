**Chatbot with LangChain and Hugging Face**
This project demonstrates how to create an interactive chatbot using LangChain and Hugging Face libraries. The chatbot leverages a pre-trained model for text generation and embeddings for document retrieval.

**Table of Contents**
Prerequisites
Installation
Setup
Running the Chatbot
Code Overview

**Prerequisites**
Ensure you have Python installed (preferably Python 3.8 or higher).

**Installation**
To set up the environment and install the required packages, follow these steps:

**Install Required Packages**
pip install -qqq flask-ngrok pyngrok==4.1.1 langchain==0.0.228 chromadb==0.3.26 sentence-transformers==2.2.2 einops==0.6.1 unstructured==0.8.0 transformers==4.30.2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install auto-gptq --no-build-isolation --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

Setup
**Obtain API Token**
Get your Hugging Face API token from Hugging Face. When prompted, enter this token to configure access to the model.

**Prepare PDF Documents**
Place your PDF documents in the ./corpus/ directory. The script will process these documents to create a vector store for searching relevant information.

**Running the Chatbot**
Run the Script
Execute the script to start the chatbot:

Copy code
python chatbot.py

**Interact with the Chatbot**
The chatbot will prompt you to enter your questions. Type your questions and press Enter. The chatbot will respond based on the context and the information retrieved from the loaded documents. To end the session, type "bye" or "goodbye".

**Code Overview**
Imports and Initialization
The script imports necessary libraries, sets up environment variables, and configures the model and tokenizer.

Model and Tokenizer
Loads the TheBloke/Nous-Hermes-13B-GPTQ model for text generation and embaas/sentence-transformers-multilingual-e5-base for embeddings.

Generation Configuration
Configures parameters for text generation such as temperature, top_p, and repetition_penalty.

Embeddings
Uses pre-trained embeddings to facilitate document search.

Document Loading
Loads and processes PDF documents into a format suitable for similarity search.

Chatbot Class
The Chatbot class sets up the conversational chain and manages interactions with users.

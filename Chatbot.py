import streamlit as st
import ollama
import PyPDF2

st.title("💬 Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
if "pdf_text" not in st.session_state:
    st.session_state["pdf_text"] = ""

### File uploader for PDF
pdf_file = st.file_uploader("Upload a PDF file", type="pdf")

### Load PDF text
if pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()
    st.session_state["pdf_text"] = pdf_text
    st.write("PDF loaded successfully. You can now start asking questions.")

### Write Message History
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message(msg["role"], avatar="🧑‍💻").write(msg["content"])
    else:
        st.chat_message(msg["role"], avatar="🤖").write(msg["content"])

## Generator for Streaming Tokens
def generate_response():
    response = ollama.chat(model='gemma:2b', stream=True, messages=st.session_state.messages)
    for partial_resp in response:
        token = partial_resp["message"]["content"]
        st.session_state["full_message"] += token
        yield token

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🧑‍💻").write(prompt)

    # Check if the prompt can be answered based on the PDF content
    if st.session_state["pdf_text"]:
        # Add the PDF content to the messages for context
        st.session_state.messages.append({"role": "system", "content": st.session_state["pdf_text"]})

    st.session_state["full_message"] = ""
    st.chat_message("assistant", avatar="🤖").write_stream(generate_response)
    if st.session_state["full_message"].strip() == "":
        st.chat_message("assistant", avatar="🤖").write("Sorry, I couldn't understand the question. I urge you to please contact the administrator or the company.")
    else:
        st.session_state.messages.append({"role": "assistant", "content": st.session_state["full_message"]})

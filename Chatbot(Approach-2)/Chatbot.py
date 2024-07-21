# Install necessary packages
!pip install -qqq flask-ngrok pyngrok==4.1.1 pip langchain==0.0.228 chromadb==0.3.26 sentence-transformers==2.2.2 einops==0.6.1 unstructured==0.8.0 transformers==4.30.2
!pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install auto-gptq --no-build-isolation --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/ 

from auto_gptq import AutoGPTQForCausalLM
from pathlib import Path
import torch
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, GenerationConfig, TextStreamer, pipeline
import os
from getpass import getpass

# Define directories and API token
HUGGINGFACEHUB_API_TOKEN = getpass()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN

# Define device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load model and tokenizer
model_id = "TheBloke/Nous-Hermes-13B-GPTQ"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(
    model_id,
    use_safetensors=True,
    trust_remote_code=True,
    device=DEVICE,
)

# Define generation configuration
generation_config = GenerationConfig.from_pretrained(model_id)
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True, use_multiprocessing=False)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15,
    generation_config=generation_config,
    streamer=streamer,
    batch_size=1
)
llm = HuggingFacePipeline(pipeline=pipe)

# Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name='embaas/sentence-transformers-multilingual-e5-base',
    model_kwargs={'device': DEVICE},
)

# Load PDF documents and create vector store
def load_pdf_documents(directory_path: Path) -> Chroma:
    loader = DirectoryLoader(directory_path, glob="**/*pdf")  # Load PDF files
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return Chroma.from_documents(texts, embeddings)

# Initialize vector store with documents from './corpus/'
db = load_pdf_documents('./corpus/')

# Define prompt template
DEFAULT_TEMPLATE = """
### Instruction: You're a support agent that is talking to a customer. Use only the chat history and the following information
{context}
to answer in a helpful manner to the question. If you don't know the answer - say that you don't know. Keep your replies short, compassionate and informative.
{chat_history}
### Input: {question}
### Response:
""".strip()

class Chatbot:
    def __init__(
        self,
        text_pipeline: HuggingFacePipeline,
        embeddings: HuggingFaceEmbeddings,
        documents_dir: Path,
        prompt_template: str = DEFAULT_TEMPLATE,
        verbose: bool = False
    ):
        prompt = PromptTemplate(
            input_variables=["context", "question", "chat_history"],
            template=prompt_template,
        )
        self.chain = self._create_chain(text_pipeline, prompt, verbose)
        self.db = self._embed_data(documents_dir, embeddings)

    def _create_chain(
        self,
        text_pipeline: HuggingFacePipeline,
        prompt: PromptTemplate,
        verbose: bool = False,
    ):
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            human_prefix='### Input',
            ai_prefix='### Response',
            input_keys='question',
            output_key='output_text',
            return_messages=False,
        )
        return load_qa_chain(
            text_pipeline,
            chain_type='stuff',
            prompt=prompt,
            memory=memory,
            verbose=verbose,
        )

    def _embed_data(self, documents_dir: Path, embeddings: HuggingFaceEmbeddings) -> Chroma:
        return load_pdf_documents(documents_dir)

    def __call__(self, user_input: str) -> str:
        docs = self.db.similarity_search(user_input)
        return self.chain.run(user_input)

# Initialize chatbot and interact
chatbot = Chatbot(llm, embeddings, './Corpus.pdf')

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

while True:
    user_input = input('You: ')
    if user_input.lower() in ["bye", "goodbye"]:
        break
    answer = chatbot(user_input)
    print(f"AI: {answer}")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from huggingface_hub import login
from langchain.chains import create_retrieval_chain


class Quantized_Phi3:
    def __init__(self, model_name):
        login(token="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")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.bnb_config = BitsAndBytesConfig(load_in_4bit=True)

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="cuda",
            cache_dir="./model_docs",
            low_cpu_mem_usage=True,
            quantization_config=self.bnb_config
        )

    def create_pipeline(self):
        self.load_model()
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="cpu",
            max_new_tokens=500,
            do_sample=True,
            top_k=3,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return pipe


class WaterXChatBot:
    def __init__(self, model_name):
        self.phi3_model = Quantized_Phi3(model_name)
        self.phi3_pipeline = self.phi3_model.create_pipeline()
        self.dataset = "waterx_dataset_2.pdf"
        self.generation_args = {
            "temperature": 0,
            "return_full_text": False,
        }

    def setup_llm(self):
        llm_model = HuggingFacePipeline(pipeline=self.phi3_pipeline, model_kwargs=self.generation_args)
        return llm_model

    def setup_custom_prompt(self):
        # Set up system prompt
        system_prompt = (
            "You are an assistant for helping in water techniques and method to solve water scarcity. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know."
            "\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        return prompt

    def dataset_loader(self):
        loader = PyPDFLoader(self.dataset)
        docs = loader.load()
        txt_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = txt_splitter.split_documents(docs)
        return final_docs

    def generate_vector_store(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_db = FAISS.from_documents(self.dataset_loader(), embeddings)
        return vector_db

    def setup_retrieval_chain(self):
        chain = create_stuff_documents_chain(self.setup_llm(), self.setup_custom_prompt())
        retriever = self.generate_vector_store().as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, chain)
        return retrieval_chain

    def generate_responses(self, inp):
        # Invoke the retrieval chain to get the answer
        response = self.setup_retrieval_chain().invoke({
            "input": inp
        })

        # Extract the human input and the assistant's response
        human_question = response['input']  # This is the original question
        assistant_answer = response['answer']  # This includes the assistant's output

        # Clean the assistant's answer to focus on the response
        assistant_answer = assistant_answer.split("Assistant:")[-1].strip()  # Extract text after "Assistant:"

        # Format the output as desired
        return f"Human: {human_question}\nAssistant: {assistant_answer}"


if __name__ == "__main__":
    model_name = "dekuthenerd/Phi-3.5-mini-instruct-bnb-4bit"
    chatbot = WaterXChatBot(model_name)

    print("Welcome to WaterX ChatBot!")
    print("Ask your questions about water techniques. Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        response = chatbot.generate_responses(user_input)
        print(response)

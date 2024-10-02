import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from langchain import HuggingFacePipeline, LLMChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from huggingface_hub import login


class Quantized_Phi3:
    def __init__(self, model_name):
        login(token="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")
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cpu")

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            cache_dir="./model_docs",
            low_cpu_mem_usage=True,
            # quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            #                                        bnb_4bit_quant_type="nf4",
            #                                        bnb_4bit_use_double_quant=True,
            #                                        bnb_4bit_compute_dtype=torch.bfloat16)
        )

    def create_pipeline(self):
        self.load_model()
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="cpu",
            max_length=600,
            do_sample=True,
            top_k=3,
            num_return_sequence=1,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return pipe


class WaterXChatBot:
    def __init__(self, model_name, inp):
        self.phi3_model = Quantized_Phi3(model_name)
        self.phi3_pipeline = self.phi3_model.create_pipeline()
        self.dataset = "waterx_agr.pdf"
        self.inp = inp

    def setup_llm(self):
        llm_model = HuggingFacePipeline(pipeline=self.phi3_pipeline)
        return llm_model

    def setup_custom_prompt(self):
        prompt = ChatPromptTemplate.from_template("""
                    Answer the following question based only on the provided context. 
                    Think step by step before providing a detailed answer. 
                    I will tip you $1000 if the user finds the answer helpful. 
                    <context>
                    {context}
                    </context>
                    Question: {input}""")
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
        chain = LLMChain(llm=self.setup_llm(), prompt=self.setup_custom_prompt())
        retriever = self.generate_vector_store().as_retriever()

        retrieval_chain = create_retrieval_chain(retriever, chain)

        return retrieval_chain

    def generate_responses(self):
        response = self.setup_retrieval_chain().invoke({
            "input": self.inp
        })

        return response['answer']


if __name__ == "__main__":
    model_name = "dekuthenerd/Phi-3.5-mini-instruct-bnb-4bit"
    # model_name = "dekuthenerd/Mistral-7B-Instruct-v0.3-autoround-4bit-sym"
    query = "what water techniques should i use for Agrciulture?"
    qa = WaterXChatBot(model_name, query)
    print(qa.generate_responses())

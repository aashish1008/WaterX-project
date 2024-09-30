import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain import HuggingFacePipeline, PromptTemplate, LLMChain


class Quantized_Phi3:
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map="auto",
            cache_dir="./model_docs"
        )

    def create_pipeline(self):
        self.load_model()
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
            max_length=600,
            do_sample=True,
            top_k=3,
            num_return_sequence=1,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return pipe


class WaterXChatBot:
    def __init__(self, model_name):
        self.phi3_model = Quantized_Phi3(model_name)
        self.phi3_pipeline = self.phi3_model.create_pipeline()

    def setup_llm(self):
        llm_model = HuggingFacePipeline(pipeline=self.phi3_pipeline)
        return llm_model



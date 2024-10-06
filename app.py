from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
from tts import TTS

load_dotenv()


class WaterXChatBot:
    def __init__(self, inp):
        self.dataset = "waterx_agr.pdf"
        self.inp = inp
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        # initialize Text-to-speech engine
        self.engine = pyttsx3.init()

    def setup_llm(self):
        llm = ChatGroq(groq_api_key=self.groq_api_key,
                       model_name="llama3-8b-8192"
                       )
        return llm

    def setup_custom_prompt(self):
        prompt = ChatPromptTemplate.from_template(
            """
            You are helpful assistant and providing solution on water scarcity issue like water efficient technique and method.
            Please provide the most accurate response based on the question
            <context>
            {context}
            <context>
            Questions:{input}
    
            """
        )

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

    def txt_to_speech(self, ans):
        self.engine.say(ans)
        # play the speech
        self.engine.runAndWait()

    def generate_responses(self):
        try:
            response = self.setup_retrieval_chain().invoke({
                "input": self.inp
            })
            return response["answer"]
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            raise


if __name__ == "__main__":
    # Test query
    query = "I have a limited budget of around INR 30,000 for implementing water-efficient techniques in agriculture. What are the most cost-effective solutions I can adopt?"

    # Initialize and run chatbot
    print("Initializing chatbot...")
    qa = WaterXChatBot(query)
    print("Generating response...")
    response = qa.generate_responses()
    print("\nQuery:", query)
    print("\nResponse:", response)
    print("Audio On...")
    qa.txt_to_speech(response)
    print("Stop..")
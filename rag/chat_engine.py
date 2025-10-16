from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import Settings, set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.memory import ChatMemoryBuffer
import os
from dotenv import load_dotenv
load_dotenv()

set_global_tokenizer(
    AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM3-3B").encode  # "HuggingFaceH4/zephyr-7b-alpha", "microsoft/Phi-3-mini-4k-instruct"
)

# print(os.getenv("HF_TOKEN"))
hf_llm = HuggingFaceInferenceAPI(
    model_name="HuggingFaceTB/SmolLM3-3B", token=os.getenv("HF_TOKEN")
)
Settings.llm = hf_llm  # Set as global LLM

def create_chat_engine(index):
    memory = ChatMemoryBuffer.from_defaults(token_limit=1000)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        llm=hf_llm,
        system_prompt=(
            "You are a chatbot named 'Meera'. Help users solve their queries in a friendly conversational manner."
            "be concise but if question repeats again elaborate response in simple english"
            "If three consecutive questions are similar, prompt the user to raise a support ticket instead"
        ),
    )
    return chat_engine

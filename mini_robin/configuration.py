# config/configuration.py
from .bedrock_llm_wrapper import BedrockLLMProvider


class RobinConfiguration:
    def __init__(self, disease_name: str,
                 num_queries: int,
                 num_assays: int,
                 num_candidates: int,
                 model_name: str = "bedrock/meta.llama3-70b-instruct-v1:0"):

        self.disease_name = disease_name
        self.num_queries = num_queries
        self.num_assays = num_assays
        self.num_candidates = num_candidates

        # Initialize the Bedrock LLM client here
        self.llm_client = BedrockLLMProvider(model_name=model_name)

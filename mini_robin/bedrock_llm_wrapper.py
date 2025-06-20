# tools/bedrock_llm_wrapper.py
import os
from crewai import LLM


class BedrockLLMProvider:
    def __init__(self, model_name, region="us-east-1"):
        self.llm = LLM(
            model=model_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            aws_region_name=region
        )

    def get_llm(self):
        return self.llm


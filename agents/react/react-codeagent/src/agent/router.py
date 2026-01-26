import os
import json
from typing import Literal
from langchain_openai import ChatOpenAI
from google.oauth2 import service_account
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm_model(provider: Literal["openai", "vertex"], model: str, temperature: float = 0):
    if provider == "vertex":
        service_account_file_path = os.environ.get("VERTEX_SERVICE_ACCOUNT", os.getcwd())
        with open(service_account_file_path, "r") as f:
            user_credentials = json.load(f)
        credentials = service_account.Credentials.from_service_account_file(
            service_account_file_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        llm = ChatGoogleGenerativeAI(
            model=model,
            credentials = credentials,
            project=user_credentials["project_id"],
            temperature=0
            )
    elif provider == "openai":
        base_url = os.environ.get("BASE_URL", None)
        api_key = os.environ.get("API_KEY")
        llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            temperature=temperature,
            api_key=api_key
        )
    else:
        raise ValueError(f"Unknown model provider: {provider}")
    return llm
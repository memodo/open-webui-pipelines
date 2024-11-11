import asyncio
import os
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

env_file_path = os.path.join(os.path.dirname(__file__), '../', '.env')

if not os.path.exists(env_file_path):
    raise FileNotFoundError(f"The file at {env_file_path} was not found.")
    
print(f"Environment file found at: {env_file_path}")
load_dotenv(env_file_path)

def printWithEmphasis(toBePrinted):
        print("**********************************************")
        print(f"{toBePrinted}")
        print("**********************************************")

client = chromadb.Client()
printWithEmphasis(client)

printWithEmphasis("Chroma Client Version")
print(client.get_version())

printWithEmphasis("Creds")
creds = os.getenv("CHROMA_SERVER_AUTHN_CREDENTIALS")
print(creds)

client2 = chromadb.HttpClient(
        host="host.docker.internal",
        port=8000,
        settings=Settings(
                chroma_client_auth_provider="chromadb.auth.token_authn.TokenAuthClientProvider",
                chroma_client_auth_credentials="chromadb-test-token",
                chroma_auth_token_transport_header="Authorization"
        )
)
printWithEmphasis(client2)

# curl -H "Authorization: Bearer chromadb-test-token" http://localhost:8000/api/v1/collections
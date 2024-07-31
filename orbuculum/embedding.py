from langchain_community.embeddings.ollama import OllamaEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    return OllamaEmbeddings(model='nomic-embed-text')

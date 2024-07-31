from orbuculum.llm.ffm import FFMEmbeddings


def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )

    # return OllamaEmbeddings(model='nomic-embed-text')
    return FFMEmbeddings(model='ffm-embedding')

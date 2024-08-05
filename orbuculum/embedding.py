from langchain_community.embeddings import OllamaEmbeddings

from orbuculum.llm.ffm import FFMEmbeddings

model_map = {
    'nomic': 'nomic-embed-text',
    'ffm': 'ffm-embedding',
}


def get_embedding_function() -> callable:
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    from orbuculum.database import orbuculum_metadata
    model = orbuculum_metadata.embedding_model

    if model == 'ffm':
        api_key = orbuculum_metadata.api_key
        return FFMEmbeddings(model='ffm-embedding', api_key=api_key)
    else:
        model_name = model_map.get(model, model)
        return OllamaEmbeddings(model=model_name)

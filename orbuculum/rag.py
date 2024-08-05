from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from orbuculum.database import CHROMA_PATH
from orbuculum.embedding import get_embedding_function
from orbuculum.llm import model_map as llm_model_map
from orbuculum.llm.ffm import FormosaFoundationModel

console = Console()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_orbuculum(query_text: str, model: str = None) -> str:
    # Prepare DB
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search DB
    results = db.similarity_search(query_text, k=5)

    # Prepare Prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # Asking LLM
    model_name = llm_model_map.get(model, model)
    if model == 'llama3-ffm':
        from orbuculum.database import orbuculum_metadata
        api_key = orbuculum_metadata.api_key
        if api_key is None:
            raise ValueError('API Key is required for llama3-ffm model.')
        llm = FormosaFoundationModel(model=model_name, ffm_api_key=api_key)
    else:
        llm = Ollama(model=model_name)
    response_answer = llm.invoke(prompt)

    return response_answer

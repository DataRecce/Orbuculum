from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from orbuculum.database import CHROMA_PATH
from orbuculum.embedding import get_embedding_function
from orbuculum.llm.ffm import FormosaFoundationModel

console = Console()

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_orbuculum(query_text: str) -> str:
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
    # model = Ollama(model="mistral")
    model = FormosaFoundationModel(model='llama3-ffm-70b-chat')
    kwargs = {
        'messages': [
            {'role': 'user', 'content': prompt}
        ]
    }
    # response_answer = model.invoke(prompt, **kwargs)
    response_answer = ""
    for token in model.stream("", kwargs=kwargs):
        response_answer += token

    return response_answer

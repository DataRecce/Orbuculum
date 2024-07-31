from langchain_chroma import Chroma
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from rich.console import Console

from orbuculum.database import CHROMA_PATH
from orbuculum.embedding import get_embedding_function

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
    model = Ollama(model="mistral")
    response_answer = model.invoke(prompt)

    return response_answer

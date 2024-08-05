import json
import os
import shutil
from typing import Optional

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from rich.console import Console

from orbuculum.embedding import get_embedding_function

CHROMA_PATH = "chroma"
DATA_PATH = 'data'

console = Console()


def load_documents(path: str = DATA_PATH) -> list[Document]:
    abs_path = os.path.abspath(path)
    console.print(f'Loading documents from: {abs_path}')
    document_loader = PyPDFDirectoryLoader(path)
    return document_loader.load()


def split_documents(documents: list[Document], chunk_size: int = 800, chunk_overlap: int = 80) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=get_embedding_function(),
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    existing_items = db.get(include=[])
    existing_ids = set(existing_items['ids'])
    console.print(f'#Existing Items: {len(existing_ids)}')

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata['id'] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks) == 0:
        console.print('No new documents to add.')
    else:
        console.print(f'Adding {len(new_chunks)} new documents.')
        new_chunk_ids = [chunk.metadata['id'] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)


def calculate_chunk_ids(chunks: list[Document]):
    # format data/xxx.pdf:a:b
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get('source')
        page = chunk.metadata.get('page')
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f'{current_page_id}:{current_chunk_index}'
        last_page_id = current_page_id

        chunk.metadata['id'] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
        console.print('Database cleared.')


class Metadata(BaseModel):
    metadata_path: str = os.path.join(CHROMA_PATH, '.metadata.json')
    embedding_model: str = ''
    api_key: Optional[str] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.load()

    def load(self):
        if not os.path.exists(self.metadata_path):
            self.embedding_model = ''
            self.api_key = None
            return
        with open(self.metadata_path, 'r') as f:
            metadata = json.load(f)
            self.embedding_model = metadata.get('embedding_model')
            self.api_key = metadata.get('api_key')

    def save(self):
        metadata = {
            'embedding_model': self.embedding_model,
        }
        if self.api_key:
            metadata['api_key'] = self.api_key

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f)


orbuculum_metadata = Metadata()

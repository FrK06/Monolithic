import annoy
from typing import List
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings

class VectorStore:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.index = annoy.AnnoyIndex(1536, 'dot')
        self.documents: List[Document] = []

    def create_index(self, dimension: int):
        self.index = annoy.AnnoyIndex(dimension, 'dot')

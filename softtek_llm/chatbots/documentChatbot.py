import tempfile
import os
import uuid
from softtek_llm.cache import Cache
from softtek_llm.exceptions import InvalidPrompt
from softtek_llm.memory import Memory, WindowMemory
from softtek_llm.models import LLMModel
from softtek_llm.schemas import Filter, Message, Response, Vector
from softtek_llm.vectorStores import VectorStore
from softtek_llm.chatbots.chatbot import Chatbot
from softtek_llm.embeddings import EmbeddingsModel
from softtek_llm.utils import strip_accents_and_special_characters
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain.document_loaders.base import BaseLoader

from typing import Dict, List, Sequence, Literal, Tuple


class DocumentChatBot(Chatbot):
    __extension_mapper = {
        "pdf": PyPDFLoader,
        "docx": Docx2txtLoader,
        "txt": TextLoader,
        "doc": Docx2txtLoader,
    }

    def __init__(
        self,
        model: LLMModel,
        knowledge_base: VectorStore,
        embeddings_model: EmbeddingsModel,
        description: str | None = None,
        memory: Memory = WindowMemory(window_size=10),
        non_valid_response: str | None = None,
        filters: List[Filter] | None = None,
        cache: Cache | None = None,
        cache_probability: float = 0.5,
        verbose: bool = False,
    ):
        super().__init__(
            model,
            description,
            memory,
            non_valid_response,
            filters,
            cache,
            cache_probability,
            verbose,
        )

        self.__knowledge_base = knowledge_base
        self.__embeddings_model = embeddings_model

    @property
    def knowledge_base(self) -> VectorStore:
        return self.__knowledge_base

    @property
    def embeddings_model(self) -> EmbeddingsModel:
        return self.__embeddings_model

    def __get_document_name_and_file_path(
        self, file: str | bytes, file_type: Literal["pdf", "doc", "docx", "txt"]
    ) -> Tuple[str, str]:
        # * Check valid file_type
        if file_type not in self.__extension_mapper.keys():
            raise ValueError(
                f"file_type must be one of {self.__extension_mapper.keys()}"
            )

        # * Read file
        if isinstance(file, str):
            if not os.path.exists(file):
                raise FileNotFoundError(f"{file} does not exists")
            document_name = ".".join(os.path.basename(file).split(".")[:-1])
            file_path = file
        elif isinstance(file, bytes):
            if document_name is None:
                raise ValueError("document_name must be provided, unless file is str")

            temporal_dir = tempfile.gettempdir()
            file_path = os.path.join(temporal_dir, f"{str(uuid.uuid4())}.{file_type}")
            with open(file_path, "wb") as f:
                f.write(file)
        else:
            raise TypeError("file must be str or bytes")

        return document_name, file_path

    def __split_document(
        self, file_type: Literal["pdf", "doc", "docx", "txt"], file_path: str
    ) -> List[str]:
        loader: BaseLoader = self.__extension_mapper[file_type](file_path)
        file_data = loader.load_and_split()
        content = [page.page_content for page in file_data]

        return content

    def __get_vectors(
        self,
        file_type: Literal["pdf", "doc", "docx", "txt"],
        file_path: str,
        document_name: str,
    ) -> List[Vector]:
        content = self.__split_document(file_type, file_path)
        embedded_file = [self.embeddings_model.embed(page) for page in content]
        vectors = [
            Vector(
                embeddings=page,
                id=f"{strip_accents_and_special_characters(document_name)}_{i}",
                metadata={"source": f"{document_name}_{i}", "text": content[i]},
            )
            for i, page in enumerate(embedded_file)
        ]

        return vectors

    def add_document(
        self,
        file: str | bytes,
        file_type: Literal["pdf", "doc", "docx", "txt"],
        document_name: str | None = None,
        knowledge_base_namespace: str | None = None,
    ):
        document_name, file_path = self.__get_document_name_and_file_path(
            file, file_type
        )
        vectors = self.__get_vectors(file_type, file_path, document_name)

        self.knowledge_base.add(vectors, namespace=knowledge_base_namespace)

    def delete_document(
        self,
        file: str | bytes,
        file_type: Literal["pdf", "doc", "docx", "txt"],
        document_name: str | None = None,
        knowledge_base_namespace: str | None = None,
    ):
        document_name, file_path = self.__get_document_name_and_file_path(
            file, file_type
        )
        vector_count = len(self.__split_document(file_type, file_path))
        self.knowledge_base.delete(
            [
                f"{strip_accents_and_special_characters(document_name)}_{i}"
                for i in range(vector_count)
            ],
            namespace=knowledge_base_namespace,
        )

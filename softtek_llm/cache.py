from datetime import datetime
from time import perf_counter_ns
from typing import Dict, Tuple
from uuid import uuid1

from softtek_llm.embeddings import EmbeddingsModel
from softtek_llm.schemas import Message, Response, Vector
from softtek_llm.vectorStores import VectorStore


class Cache:
    """
    # Cache
    Represents the cache of the assistant. Stores all the prompts and responses that have been exchanged between the user and the assistant.

    ## Attributes
    - `vector_store` (VectorStore): The vector store used to store the prompts and responses.
    - `embeddings_model` (EmbeddingsModel): The embeddings model used to generate embeddings for prompts.

    ## Methods
    - `add`: Adds a prompt and response to the cache. It calculates the embeddings for the prompt and adds it to the vector store.
    - `retrieve`: Retrieves the best response from a query using the prompt provided by the user. It calculates the time taken to retrieve the data and returns the response.
    """

    def __init__(self, vector_store: VectorStore, embeddings_model: EmbeddingsModel):
        """Initializes the Cache class.

        Args:
            vector_store (VectorStore): The vector store used to store the prompts and responses.
            embeddings_model (EmbeddingsModel): The embeddings model used to generate embeddings for prompts.
        """
        self.vector_store = vector_store
        self.embeddings_model = embeddings_model

    @property
    def vector_store(self) -> VectorStore:
        """The vector store used to store the prompts and responses."""
        return self.__vector_store

    @vector_store.setter
    def vector_store(self, vector_store: VectorStore):
        if not isinstance(vector_store, VectorStore):
            raise TypeError("vector_store must be of type VectorStore or a subclass.")
        self.__vector_store = vector_store

    @property
    def embeddings_model(self) -> EmbeddingsModel:
        """The embeddings model used to generate embeddings for prompts."""
        return self.__embeddings_model

    @embeddings_model.setter
    def embeddings_model(self, embeddings_model: EmbeddingsModel):
        if not isinstance(embeddings_model, EmbeddingsModel):
            raise TypeError(
                "embeddings_model must be of type EmbeddingsModel or a subclass."
            )
        self.__embeddings_model = embeddings_model

    def add(self, prompt: str, response: Response, **kwargs):
        """This function adds a prompt and response to the cache. It calculates the embeddings for the prompt and adds it to the vector store.

        Args:
            prompt (str): A string prompt to which the function will respond to.
            response (Response): A Response object containing the model's reply, timestamp, latency, and model name.
        """
        vector = Vector(
            embeddings=self.embeddings_model.embed(prompt, **kwargs),
            id=str(uuid1()),
            metadata={
                "response": response.message.content,
                "model": response.model,
            },
        )

        self.vector_store.add([vector], **kwargs)

    def retrieve(
        self,
        prompt: str,
        threshold: float = 0.9,
        additional_kwargs: Dict = {},
        **kwargs
    ) -> Tuple[Response | None, float]:
        """This function retrieves the best response from a query using the prompt provided by the user. It calculates the time taken to retrieve the data and returns the response.

        Args:
            prompt (str): A string prompt to which the function will respond to.
            threshold (float, optional): The threshold to use for the search. Defaults to 0.9.
            additional_kwargs (Dict, optional): Optional dictionary of additional keyword arguments to add to the retrieved response. Defaults to {}.

        Returns:
            Tuple[Response | None, float]: A tuple containing the response and the score of the best match.
        """
        start = perf_counter_ns()
        prompt_vector = Vector(
            embeddings=self.embeddings_model.embed(prompt, **kwargs),
        )

        matches = sorted(
            self.vector_store.search(prompt_vector, **kwargs),
            key=lambda x: x.metadata["score"],
            reverse=True,
        )

        if len(matches) == 0:
            return None, 0.0

        best_match = matches[0]

        if best_match.metadata["score"] < threshold:
            return None, best_match.metadata["score"]

        return (
            Response(
                message=Message(
                    role="assistant", content=best_match.metadata["response"]
                ),
                created=int(datetime.utcnow().timestamp()),
                latency=int((perf_counter_ns() - start) / 1e6),
                from_cache=True,
                model=best_match.metadata["model"],
                additional_kwargs=additional_kwargs,
            ),
            best_match.metadata["score"],
        )

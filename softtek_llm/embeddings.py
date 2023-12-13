"""
# Embeddings
This module contains classes for embeddings models.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal

import openai
import requests
from typing_extensions import override

from softtek_llm.utils import setup_azure


class EmbeddingsModel(ABC):
    """
    # Embeddings Model
    Creates an abstract base class for an embeddings model. Used as a base class for implementing different types of embeddings models.

    ## Methods
    - `embed`: Method to embed text. Must be implemented by the child class.
    """

    def __init__(self, **kwargs: Any):
        """Initializes the EmbeddingsModel class."""
        super().__init__()

    @abstractmethod
    def embed(self, prompt: str, **kwargs: Any) -> List[float]:
        """
        This is an abstract method for embedding a prompt into a list of floats. This method must be implemented by a subclass.

        Args:
        - `prompt` (str): The string prompt to embed.
        - `**kwargs` (Any): Additional arguments for implementation-defined use.

        Returns:
        - (List[float]): The embedding of the prompt as a list of floats.

        Raises:
        - NotImplementedError: When this abstract method is called without being implemented in a subclass.
        """
        raise NotImplementedError("embed method must be overridden")


class OpenAIEmbeddings(EmbeddingsModel):
    """
    # OpenAI Embeddings
    Creates an OpenAI embeddings model. This class is a subclass of the EmbeddingsModel abstract base class.

    ## Attributes
    - `model_name`: Embeddings model name.

    ## Methods
    - `embed`: Embeds a prompt into a list of floats.
    """

    @override
    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_type: Literal["azure"] | None = None,
        api_base: str | None = None,
        api_version: str = "2023-07-01-preview",
    ):
        """Initializes the OpenAIEmbeddings class.

        Args:
            `api_key` (str): OpenAI API key.
            `model_name` (str): OpenAI embeddings model name.
            `api_type` (Literal["azure"] | None, optional): Type of API to use. Defaults to None.
            `api_base` (str | None, optional): Base URL for Azure API. Defaults to None.
            `api_version` (str, optional): API version for Azure API. Defaults to "2023-07-01-preview".

        Raises:
            (ValueError): When api_type is not "azure" or None.
        """
        super().__init__()
        openai.api_key = api_key
        self.__model_name = model_name

        if api_type is not None:
            openai.api_type = api_type
            match api_type:
                case "azure":
                    setup_azure(api_base, api_version)
                case _:
                    raise ValueError(
                        f"api_type must be either 'azure' or None, not {api_type}"
                    )

    @property
    def model_name(self) -> str:
        """Embeddings model name."""
        return self.__model_name

    @override
    def embed(self, prompt: str, **kwargs) -> List[float]:
        """Embeds a prompt into a list of floats.

        Args:
            `prompt` (str): Prompt to embed.

        Returns:
            (List[float]): Embedding of the prompt as a list of floats.
        """
        response = openai.Embedding.create(
            deployment_id=self.model_name,
            input=prompt,
        )
        return response["data"][0]["embedding"]


class SofttekOpenAIEmbeddings(EmbeddingsModel):
    """
    # Softtek OpenAI Embeddings
    Creates a Softtek OpenAI embeddings model. This class is a subclass of the EmbeddingsModel abstract base class.

    ## Attributes
    - `model_name`: Embeddings model name.
    - `api_key`: API key for the Softtek OpenAI API.

    ## Methods
    - `embed`: Embeds a prompt into a list of floats.
    """

    @override
    def __init__(self, model_name: str, api_key: str):
        """Initializes the SofttekOpenAIEmbeddings class.

        Args:
            `model_name` (str): Name of the embeddings model.

            `api_key` (str): API key for the Softtek OpenAI API.
        """
        super().__init__()
        self.__model_name = model_name
        self.__api_key = api_key

    @property
    def model_name(self) -> str:
        """Embeddings model name."""
        return self.__model_name

    @override
    def embed(self, prompt: str, additional_kwargs: Dict = {}) -> List[float]:
        """Embeds a prompt into a list of floats.

        Args:
            `prompt` (str): Prompt to embed.

            `additional_kwargs` (Dict, optional): Additional keyword arguments. Defaults to {}.

        Returns:
            `List[float]`: Embedding of the prompt as a list of floats.

        Raises:
            (Exception): When the API returns a non-200 status code.
        """
        response = requests.post(
            "https://llm-api-stk.azurewebsites.net/embeddings",
            headers={"api-key": self.__api_key},
            json={
                "input": prompt,
                "model": self.model_name,
                "additional_kwargs": additional_kwargs,
            },
        )

        if response.status_code != 200:
            raise Exception(response.json()["detail"])

        return response.json()["data"][0]["embedding"]

from abc import ABC, abstractmethod
from time import perf_counter_ns
from typing import Any, Dict, List, Literal

import openai
import requests
from openai.error import InvalidRequestError
from typing_extensions import override

from softtek_llm.exceptions import TokensExceeded
from softtek_llm.memory import Memory
from softtek_llm.schemas import Filter, Message, OpenAIChatResponse, Response
from softtek_llm.utils import setup_azure


class LLMModel(ABC):
    """
    # LLM Model
    Creates an abstract base class for a language model. Used as a base class for implementing different types of language models. Provides intialization with options like max_tokens, temperature and name. Also defines a call method that must be implemented.

    ## Parameters
    - `name`: Name of the model

    ## Methods
    - `__call__`: Method to generate text from the model. Must be implemented by the child class.
    """

    def __init__(
        self,
        model_name: str,
        verbose: bool = False,
        **kwargs: Any,
    ):
        """Initializes the LLMModel class.

        Args:
            `model_name` (str): Name of the model
            `verbose` (bool, optional): Whether to print debug messages. Defaults to False.
        """
        super().__init__()
        self.__model_name = model_name
        self.__verbose = verbose

    @property
    def model_name(self) -> str:
        """Tthe name of the model."""
        return self.__model_name

    @property
    def verbose(self) -> bool:
        """Whether to print debug messages."""
        return self.__verbose

    @abstractmethod
    def __call__(
        self, memory: Memory, description: str = "You are a bot", **kwargs: Any
    ) -> Response:
        """
        A method to be overridden that calls the model to generate text.

        Args:
            `memory` (Memory): An instance of the Memory class containing the conversation history.
            `description` (str, optional): Description of the model. Defaults to "You are a bot".

        Returns:
            Response: The generated response.

        Raises:
        - NotImplementedError: When this abstract method is called without being implemented in a subclass.
        """
        raise NotImplementedError("__call__ method must be overridden")

    @abstractmethod
    def parse_filters(self, prompt: str) -> List[Message]:
        """
        Generates a prompt message to check if a given prompt follows a set of filtering rules.

        Args:
            `prompt` (str): a string representing the prompt that will be checked against rules

        Raises:
         - NotImplementedError: When this abstract method is called without being implemented in a subclass.
        """
        raise NotImplementedError("parse_filters method must be overriden")


class OpenAI(LLMModel):
    """
    # OpenAI
    Creates an OpenAI language model. This class is a subclass of the LLMModel abstract base class.

    ## Attributes
    - `model_name`: Language model name.
    - `max_tokens`: The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length.
    - `temperature`: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    - `presence_penalty`: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    - `frequency_penalty`: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.

    ## Methods
    - `__call__`: Method to generate text from the model.
    """

    @override
    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_type: Literal["azure"] | None = None,
        api_base: str | None = None,
        api_version: str = "2023-09-01-preview",
        max_tokens: int | None = None,
        temperature: float = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        verbose: bool = False,
    ):
        """Initializes the OpenAI LLM Model class.

        Args:
            `api_key` (str): OpenAI API key.
            `model_name` (str): Name of the model. If you're using an OpenAI resource/key, use the correspondent model name (e.g. gpt-35-turbo-16k), if you're using Azure OpenAI or other hosting service, use your deployment name.
            `api_type` (Literal["azure"] | None, optional): Type of API to use. Defaults to None.
            `api_base` (str | None, optional): Base URL for Azure API. Defaults to None.
            `api_version` (str, optional): API version for Azure API. Defaults to "2023-07-01-preview".
            `max_tokens` (int | None, optional): The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length. Defaults to None.
            `temperature` (float, optional): What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Defaults to 1.
            `presence_penalty` (float, optional): Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Defaults to 0.
            `frequency_penalty` (float, optional): Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Defaults to 0.
            `verbose` (bool, optional): Whether to print debug messages. Defaults to False.

        Raises:
            ValueError: When api_type is not "azure" or None.
        """
        super().__init__(model_name, verbose=verbose)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        openai.api_key = api_key
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
    def max_tokens(self) -> int | None:
        """The maximum number of tokens to generate in the chat completion.

        The total length of input tokens and generated tokens is limited by the model's context length.
        """
        return self.__max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int | None):
        if not isinstance(value, int) and value is not None:
            raise TypeError(
                f"max_tokens must be an integer or None, not {type(value).__name__}"
            )
        self.__max_tokens = value

    @property
    def temperature(self) -> float:
        """
        What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        """
        return self.__temperature

    @temperature.setter
    def temperature(self, value: float):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(f"temperature must be a float, not {type(value).__name__}")

        if value < 0 or value > 2:
            raise ValueError("temperature must be between 0 and 2")
        self.__temperature = value

    @property
    def presence_penalty(self) -> float:
        """
        Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        """
        return self.__presence_penalty

    @presence_penalty.setter
    def presence_penalty(self, value: float):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(
                f"presence_penalty must be a float, not {type(value).__name__}"
            )
        if value < -2 or value > 2:
            raise ValueError("presence_penalty must be between -2 and 2")
        self.__presence_penalty = value

    @property
    def frequency_penalty(self) -> float:
        """
        Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        """
        return self.__frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value: float):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(
                f"frequency_penalty must be a float, not {type(value).__name__}"
            )
        if value < -2 or value > 2:
            raise ValueError("frequency_penalty must be between -2 and 2")
        self.__frequency_penalty = value

    @override
    def __call__(self, memory: Memory, description: str = "You are a bot.") -> Response:
        """
        Process a conversation using the OpenAI model and return a Response object.

        This function sends a conversation stored in the 'memory' parameter to the specified OpenAI model
        (self.model_name), retrieves a response from the model, and records the conversation in 'memory'.
        It then constructs a Response object containing the model's reply.

        Args:
            `memory` (Memory): An instance of the Memory class containing the conversation history.
            `description` (str, optional): Description of the model. Defaults to "You are a bot.".

        Returns:
            `resp` (Response): A Response object containing the model's reply, timestamp, latency, and model name.
        """

        start = perf_counter_ns()
        messages = [message.model_dump() for message in memory.get_messages()]
        messages.insert(0, Message(role="system", content=description).model_dump())

        if self.verbose:
            print(f"Memory: {messages}")

        try:
            answer = OpenAIChatResponse(
                **openai.ChatCompletion.create(
                    deployment_id=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    presence_penalty=self.presence_penalty,
                    frequency_penalty=self.frequency_penalty,
                )
            )
        except InvalidRequestError as e:
            if "maximum context length" in str(e).lower():
                raise TokensExceeded(
                    f"Tokens exceeded for model '{self.model_name}'. If you're using the max_tokens parameter, try increasing it. Otherwise, you may consider:\n- Upgrading to a different model\n- Reducing the messages stored in memory (for example, by using a WindowMemory)\n- Applying some strategy for data reduction (for example, text summarization)"
                ) from e
            else:
                raise e

        resp = Response(
            message=Message(
                role="assistant", content=answer.choices[0].message.content
            ),
            created=answer.created,
            latency=(perf_counter_ns() - start) // 1e6,
            from_cache=False,
            model=answer.model,
            usage=answer.usage,
        )

        memory.add_message(resp.message.role, resp.message.content)

        return resp

    @override
    def parse_filters(
        self, prompt: str, context: List[Message], filters: List[Filter]
    ) -> List[Message]:
        """
        Generates a prompt message to check if a given prompt follows a set of filtering rules.

        Args:
            `prompt` (str): a string representing the prompt that will be checked against rules.
            `context` (List[Message]): A list containing the last 3 messages from the chat.
            `filters` (List[Filter]): List of filters used by the chatbot.

        Returns:
            (List[Message]): a list of messages to be used by the chatbot to check if the prompt respects the rules
        """
        context = "\n".join(
            [f"\t- {message.role}: {message.content}" for message in context]
        )

        rules = "\n".join([f"\t- {filter.type}: {filter.case}" for filter in filters])

        prompt = f'Considering this context:\n{context}\n\nPlease review the prompt below and answer with "yes" if it adheres to the rules or "no" if it violates any of the rules.\nRules:\n{rules}\n\nPrompt:\n{prompt}'

        if self.verbose:
            print(f"Revision prompt: {prompt}")

        messages = [
            Message(**{"role": "system", "content": "only respond with 'yes' or 'no'"}),
            Message(**{"role": "user", "content": prompt}),
        ]

        return messages


class SofttekOpenAI(LLMModel):
    """
    # Softtek OpenAI
    Creates a Softtek OpenAI language model. This class is a subclass of the LLMModel abstract base class.

    ## Attributes
    - `model_name`: Language model name.
    - `max_tokens`: The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length.
    - `temperature`: What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
    - `presence_penalty`: Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
    - `frequency_penalty`: Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
    - `logit_bias`: A map of tokens to their desired logit bias. The keys are tokens, and the values are the bias to add to the logits (before applying temperature).
    - `stop`: One or more sequences where the API will stop generating further tokens.
    - `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
    - `user`: The name of the user for which the API will generate a response.
    - `api_key`: The OpenAI API key.

    ## Methods
    - `__call__`: Method to generate text from the model.
    - `parse_filters`: Generates a prompt message to check if a given prompt follows a set of filtering rules.
    """

    @override
    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_tokens: int | None = None,
        temperature: float = 1,
        presence_penalty: float = 0,
        frequency_penalty: float = 0,
        logit_bias: Dict[int, int] | None = None,
        stop: str | List[str] | None = None,
        top_p: float = 1,
        user: str | None = None,
        verbose: bool = False,
    ):
        """Initializes the Softtek OpenAI LLM Model class.

        Args:
            `api_key` (str): LLMOPs API key.

            `model_name` (str): Name of the 
            model.

            `max_tokens` (int | None, optional): T
            he maximum number of tokens 
            to generate in the chat completion. The total length of input tokens and generated tokens is limited by the model's context length. Defaults to None.
            
            `temperature` (float, optional): What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic. Defaults to 1.

            `presence_penalty` (float, optional): Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Defaults to 0.

            `frequency_penalty` (float, optional): Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Defaults to 0.

            `logit_bias` (Dict[int, int] | None, optional): A map of tokens to their desired logit bias. The keys are tokens, and the values are the bias to add to the logits (before applying temperature). Defaults to None.

            `stop` (str | List[str] | None, optional): One or more sequences where the API will stop generating further tokens. Defaults to None.

            `top_p` (float, optional): An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. Defaults to 1.

            `user` (str | None, optional): The name of the user for which the API will generate a response. Defaults to None.

            `verbose` (bool, optional): Whether to print debug messages. Defaults to False.
        """
        super().__init__(model_name, verbose=verbose)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.logit_bias = logit_bias
        self.stop = stop
        self.top_p = top_p
        self.user = user
        self.__api_key = api_key

    @property
    def max_tokens(self) -> int | None:
        """The maximum number of tokens to generate in the chat completion.

        The total length of input tokens and generated tokens is limited by the model's context length.
        """
        return self.__max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int | None):
        if not isinstance(value, int) and value is not None:
            raise TypeError(
                f"max_tokens must be an integer or None, not {type(value).__name__}"
            )
        self.__max_tokens = value

    @property
    def temperature(self) -> float:
        """
        What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.
        """
        return self.__temperature

    @temperature.setter
    def temperature(self, value: float):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(f"temperature must be a float, not {type(value).__name__}")

        if value < 0 or value > 2:
            raise ValueError("temperature must be between 0 and 2")
        self.__temperature = value

    @property
    def presence_penalty(self) -> float:
        """
        Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
        """
        return self.__presence_penalty

    @presence_penalty.setter
    def presence_penalty(self, value: float):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(
                f"presence_penalty must be a float, not {type(value).__name__}"
            )
        if value < -2 or value > 2:
            raise ValueError("presence_penalty must be between -2 and 2")
        self.__presence_penalty = value

    @property
    def frequency_penalty(self) -> float:
        """
        Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
        """
        return self.__frequency_penalty

    @frequency_penalty.setter
    def frequency_penalty(self, value: float):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(
                f"frequency_penalty must be a float, not {type(value).__name__}"
            )
        if value < -2 or value > 2:
            raise ValueError("frequency_penalty must be between -2 and 2")
        self.__frequency_penalty = value

    @property
    def logit_bias(self) -> Dict[int, int] | None:
        """
        A map of tokens to their desired logit bias. The keys are tokens, and the values are the bias to add to the logits (before applying temperature).
        """
        return self.__logit_bias

    @logit_bias.setter
    def logit_bias(self, value: Dict[int, int] | None):
        if not isinstance(value, dict) and value is not None:
            raise TypeError(
                f"logit_bias must be a dict or None, not {type(value).__name__}"
            )
        self.__logit_bias = value

    @property
    def stop(self) -> str | List[str] | None:
        """
        One or more sequences where the API will stop generating further tokens.
        """
        return self.__stop

    @stop.setter
    def stop(self, value: str | List[str] | None):
        if (
            not isinstance(value, str)
            and not isinstance(value, list)
            and value is not None
        ):
            raise TypeError(
                f"stop must be a string, list or None, not {type(value).__name__}"
            )
        self.__stop = value

    @property
    def top_p(self) -> float:
        """
        An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered.
        """
        return self.__top_p

    @top_p.setter
    def top_p(self, value: float):
        if not isinstance(value, float) and not isinstance(value, int):
            raise TypeError(f"top_p must be a float, not {type(value).__name__}")
        if value < 0 or value > 1:
            raise ValueError("top_p must be between 0 and 1")
        self.__top_p = value

    @property
    def user(self) -> str | None:
        """
        The name of the user for which the API will generate a response.
        """
        return self.__user

    @user.setter
    def user(self, value: str | None):
        if not isinstance(value, str) and value is not None:
            raise TypeError(
                f"user must be a string or None, not {type(value).__name__}"
            )
        self.__user = value

    @property
    def api_key(self) -> str:
        """The OpenAI API key."""
        return self.__api_key

    @override
    def __call__(self, memory: Memory, description: str = "You are a bot.") -> Response:
        """
        Conversational interface that interacts with the SofttekOpenAI Chat API to generate a response.

        Args:
            `memory` (Memory): An instance of the Memory class that holds previous conversation messages.\n
            `description` (str, optional): A description of the conversation. Defaults to "You are a bot.".

        Raises:
            `Exception`: If the API request to the OpenAI Chat API returns a status code other than 200.

        Returns:
            `resp` (Response): An instance of the Response class that contains the generated response.
        """
        start = perf_counter_ns()
        messages = [message.model_dump() for message in memory.get_messages()]
        messages.insert(0, Message(role="system", content=description).model_dump())

        if self.verbose:
            print(f"Memory: {messages}")

        req = requests.post(
            "https://llm-api-stk.azurewebsites.net/chat/completions",
            headers={"api-key": self.api_key},
            json={
                "messages": messages,
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "logit_bias": self.logit_bias,
                "stop": self.stop,
                "top_p": self.top_p,
                "user": self.user,
            },
        )

        if req.status_code != 200:
            raise Exception(req.reason)

        answer = OpenAIChatResponse(**req.json())

        resp = Response(
            message=Message(
                role="assistant", content=answer.choices[0].message.content
            ),
            created=answer.created,
            latency=(perf_counter_ns() - start) // 1e6,
            from_cache=False,
            model=answer.model,
            usage=answer.usage,
        )

        memory.add_message(resp.message.role, resp.message.content)

        return resp

    @override
    def parse_filters(
        self, prompt: str, context: List[Message], filters: List[Filter]
    ) -> List[Message]:
        """
        Generates a prompt message to check if a given prompt follows a set of filtering rules.

        Args:
            `prompt` (str): a string representing the prompt that will be checked against rules.\n
            `context` (List[Message]): A list containing the last 3 messages from the chat.\n
            `filters` (List[Filter]): List of filters used by the chatbot.

        Returns:
            `messages` (List[Message]): a list of messages to be used by the chatbot to check if the prompt respects the rules
        """
        context = "\n".join(
            [f"\t- {message.role}: {message.content}" for message in context]
        )

        rules = "\n".join([f"\t- {filter.type}: {filter.case}" for filter in filters])

        prompt = f'Considering this context:\n{context}\n\nPlease review the prompt below and answer with "yes" if it adheres to the rules or "no" if it violates any of the rules.\nRules:\n{rules}\n\nPrompt:\n{prompt}'

        if self.verbose:
            print(f"Revision prompt: {prompt}")

        messages = [
            Message(**{"role": "system", "content": "only respond with 'yes' or 'no'"}),
            Message(**{"role": "user", "content": prompt}),
        ]

        return messages

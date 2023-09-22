import random
from datetime import datetime
from time import perf_counter_ns
from typing import Dict, List, Sequence

from softtek_llm.cache import Cache
from softtek_llm.exceptions import InvalidPrompt
from softtek_llm.memory import Memory, WindowMemory
from softtek_llm.models import LLMModel
from softtek_llm.schemas import Filter, Message, Response


class Chatbot:
    """
    # Chatbot
    The Chatbot class is the main class of the library. It is used to initialize a chatbot instance, which can then be used to chat with an LLM.

    ## Attributes
    - `model`: The LLM model used by the chatbot.
    - `memory`: The memory used by the chatbot.
    - `description`: Information about the chatbot.
    - `non_valid_response`: Response given when the prompt does not follow the rules set by the filters. If None, an InvalidPrompt exception is raised when the prompt does not follow the rules set by the filters.
    - `filters`: List of filters used by the chatbot.
    - `cache`: Cache used by the chatbot.
    - `cache_probability`: Probability of using the cache. If 1.0, the cache is always used. If 0.0, the cache is never used.

    ## Methods
    - `chat`: Chatbot function that returns a response given a prompt. If a memory and/or cache are available, it considers previously stored conversations. Filters are applied to the prompt before processing to ensure it is valid.
    """

    def __init__(
        self,
        model: LLMModel,
        description: str | None = None,
        memory: Memory = WindowMemory(window_size=10),
        non_valid_response: str | None = None,
        filters: List[Filter] | None = None,
        cache: Cache | None = None,
        cache_probability: float = 0.5,
        verbose: bool = False,
    ):
        """Initializes the Chatbot class.

        Args:
            model (LLMModel): LLM model used by the chatbot.
            description (str | None, optional): Information about the chatbot. Defaults to None.
            memory (Memory, optional): Memory used by the chatbot. Defaults to WindowMemory(window_size=10).
            non_valid_response (str | None, optional): Response given when the prompt does not follow the rules set by the filters. Defaults to None. If None, an InvalidPrompt exception is raised when the prompt does not follow the rules set by the filters.
            filters (List[Filter] | None, optional): List of filters used by the chatbot. Defaults to None.
            cache (Cache | None, optional): Cache used by the chatbot. Defaults to None.
            cache_probability (float, optional): Probability of using the cache. Defaults to 0.5. If 1.0, the cache is always used. If 0.0, the cache is never used.
            verbose (bool, optional): Whether to print additional information. Defaults to False.
        """
        self.model = model
        self.memory = memory
        self.description = description
        self.non_valid_response = non_valid_response
        self.filters = filters
        self.cache = cache
        self.cache_probability = cache_probability
        self.__verbose = verbose

    @property
    def model(self) -> LLMModel:
        """The model used by the chatbot."""
        return self.__model

    @model.setter
    def model(self, model: LLMModel):
        if not isinstance(model, LLMModel):
            raise TypeError("model must be of type LLMModel or a subclass.")
        self.__model = model

    @property
    def memory(self) -> Memory:
        """The memory used by the chatbot."""
        return self.__memory

    @memory.setter
    def memory(self, memory: Memory):
        if not isinstance(memory, Memory):
            raise TypeError("memory must be of type Memory or a subclass.")
        self.__memory = memory

    @property
    def description(self) -> str | None:
        """Information about the chatbot."""
        return self.__description

    @description.setter
    def description(self, description: str | None):
        if not isinstance(description, str) and description is not None:
            raise TypeError("description must be of type str or None.")
        self.__description = description if description else "You are a chatbot."

    @property
    def filters(self) -> List[Filter]:
        """The filters used by the chatbot."""
        return self.__filters

    @filters.setter
    def filters(self, filters: List[Filter] | None):
        if not isinstance(filters, Sequence) and filters is not None:
            raise TypeError("filters must be a list-like object or None.")
        if isinstance(filters, Sequence) and not all(
            isinstance(filter, Filter) for filter in filters
        ):
            raise TypeError("filters must be a list of Filter instances.")
        if isinstance(filters, Sequence) and len(filters) == 0:
            self.__filters = None
        self.__filters = filters

    @property
    def cache(self) -> Cache:
        """The cache used by the chatbot."""
        return self.__cache

    @cache.setter
    def cache(self, cache: Cache | None):
        if not isinstance(cache, Cache) and cache is not None:
            # raise TypeError explaining that cache must be of type Cache or a subclass, and showing the actual type passed
            raise TypeError(
                f"cache must be of type Cache or a subclass, but got {cache.__class__.__name__}."
            )
        self.__cache = cache

    @property
    def cache_probability(self) -> float:
        """The probability of using the cache."""
        return self.__cache_probability

    @cache_probability.setter
    def cache_probability(self, cache_probability: float):
        if not isinstance(cache_probability, float):
            raise TypeError("cache_probability must be of type float.")
        if cache_probability < 0 or cache_probability > 1:
            raise ValueError("cache_probability must be between 0 and 1 (inclusive).")
        self.__cache_probability = cache_probability

    @property
    def verbose(self) -> bool:
        """Whether to print additional information."""
        return self.__verbose

    def __random_boolean(self) -> bool:
        """Generates a random boolean value based on the given probability.

        Returns:
            bool: True if randomly generated number is less than or equal the probability, False otherwise.
        """
        return random.random() <= self.cache_probability

    def __revise(self, prompt: str) -> bool:
        """
        This method is used to revise a given prompt passed as input parameter and returns a Boolean value indicating whether the revision occurred successfully or not.

        Args:
            prompt (str): The input prompt to be revised

        Returns:
            bool: A Boolean value indicating if the revision was successful or not. True if revision was successful and, False otherwise.

        """
        revision_messages = self.model.parse_filters(prompt=prompt, context=self.memory.get_messages()[-3:], filters=self.filters)
        memory = Memory.from_messages(revision_messages)
        reviser = self.model(
            memory,
            description="As a prompt reviser, your role is to determine whether the given prompt respects the rules provided below. You can only respond with 'yes' or 'no.' The rules are absolute, meaning that if any of them are not respected, the prompt is considered invalid.",
        )

        return "yes" in reviser.message.content.lower()

    def __call_model(self) -> Response:
        """
        This method is used to call the model and returns a Response object.

        Returns:
            Response: A Response object containing the response message generated by the model.
        """
        return self.model(self.memory, description=self.description)

    def chat(
        self, prompt: str, print_cache_score: bool = False, cache_kwargs: Dict = {}
    ) -> Response:
        """
        Chatbot function that returns a response given a prompt. If a memory and/or cache are available, it considers previously stored conversations. Filters are applied to the prompt before processing to ensure it is valid.

        Args:
            prompt (str): user's input string text
            print_cache_score (bool, optional): whether to print the cache score. Defaults to False.
            cache_kwargs (Dict, optional): additional keyword arguments to be passed to the cache. Defaults to {}.

        Returns:
            last_message (Response): the response message object generated by the chatbot, including its content and metadata
        """
        start = perf_counter_ns()
        if self.filters:
            if not self.__revise(prompt):
                if self.non_valid_response:
                    return Response(
                        message=Message(role="system", content=self.non_valid_response),
                        created=int(datetime.utcnow().timestamp()),
                        latency=int((perf_counter_ns() - start) / 1e6),
                        from_cache=False,
                        model="reviser",
                    )
                raise InvalidPrompt(
                    "The prompt does not follow the rules set by the filters. If this behavior is not intended, consider modifying the filters. It is recommended to use LLMs for meta prompts."
                )

        self.memory.add_message(role="user", content=prompt)
        if not self.cache:
            last_message = self.__call_model()
        else:
            if self.__random_boolean():
                cached_response, cache_score = self.cache.retrieve(
                    prompt=prompt, **cache_kwargs
                )
                if print_cache_score:
                    print(f"Cache score: {cache_score}")
                if cached_response:
                    self.memory.add_message(
                        role=cached_response.message.role,
                        content=cached_response.message.content,
                    )
                    last_message = cached_response
                else:
                    last_message = self.__call_model()
                    self.cache.add(prompt=prompt, response=last_message, **cache_kwargs)
            else:
                last_message = self.__call_model()

        return last_message

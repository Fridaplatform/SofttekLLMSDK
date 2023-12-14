# Models

This module contains the `LLMModel` abstract base class and its implementations. These classes are used to create **language models** that can be used by [chatbots]() to generate responses.

### Index

- [LLMModel](#llmmodel)
- [OpenAI](#openai)
- [Softtek OpenAI](#softtek-openai)

## LLM Model

```python
LLMModel(
  model_name: str,
  verbose: bool = False,
  **kwargs: Any
)
```

Creates an abstract base class for a language model. Used as a base class for implementing different types of language models. Also defines a call method that must be implemented.

#### Args

- `model_name` (`str`): Name of the model
- `verbose` (`bool`, optional): Whether to print debug messages. Defaults to `False`.

### Properties

- `model_name`: The name of the model.
- `verbose`: Whether to print debug messages.

### Methods

```python
__call__(
  memory: Memory,
  description: str = "You are a bot",
  **kwargs: Any
) -> Response
```

A method to be overridden that calls the model to generate text.

#### Args

- `memory` (`Memory`): An instance of the [`Memory`]() class containing the conversation history.
- `description` (`str`, optional): Description of the model. Defaults to `"You are a bot"`.

#### Returns

- [`Response`](): The generated response.

#### Raises

- `NotImplementedError`: When this abstract method is called without being implemented in a subclass.

```python
parse_filters(
  prompt: str
) -> List[Message]
```

Generates a prompt message to check if a given prompt follows a set of filtering rules. It must be overridden in a subclass.

#### Args

- `prompt` (`str`): a string representing the prompt that will be checked against rules.

#### Returns

- `List[`[`Message`]()`]`: a list of messages to be used by the chatbot to check if the prompt respects the rules.

#### Raises

- `NotImplementedError`: When this abstract method is called without being implemented in a subclass.

## OpenAI

```python
OpenAI(
  api_key: str,
  model_name: str,
  api_type: Literal["azure"] | None = None,
  api_base: str | None = None,
  api_version: str = "2023-09-01-preview",
  max_tokens: int | None = None,
  temperature: float = 1,
  presence_penalty: float = 0,
  frequency_penalty: float = 0,
  verbose: bool = False
)
```

Creates an `OpenAI` language model. This class is a subclass of the [`LLMModel`](#llm-model) abstract base class.

#### Args

- `api_key` (`str`): OpenAI API key.
- `model_name` (`str`): Name of the model. If you're using an OpenAI resource/key, use the correspondent model name (e.g. `gpt-35-turbo-16k`), if you're using Azure OpenAI or other hosting service, use your **deployment name**.
- `api_type` (`Literal["azure"] | None`, optional): Type of API to use. Defaults to `None`.
- `api_base` (`str | None`, optional): Base URL for Azure API. Defaults to `None`.
- `api_version` (`str`, optional): API version for Azure API. Defaults to `"2023-07-01-preview"`.
- `max_tokens` (`int | None`, optional): The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is **limited by the model's context length**. Defaults to `None`.
- `temperature` (`float`, optional): What sampling temperature to use, between `0` and `2`. Higher values like `0.8` will make the output more random, while lower values like `0.2` will make it more focused and deterministic. Defaults to `1`.
- `presence_penalty` (`float`, optional): Number between `-2.0` and `2.0`. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Defaults to `0`.
- `frequency_penalty` (`float`, optional): Number between `-2.0` and `2.0`. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Defaults to `0`.
- `verbose` (`bool`, optional): Whether to print debug messages. Defaults to `False`.

#### Raises

- `ValueError`: When `api_type` is not `"azure"` or `None`.

### Properties

- `model_name`: The name of the model.
- `verbose`: Whether to print debug messages.
- `max_tokens`: The maximum number of tokens to generate in the chat completion.
- `temperature`: What sampling temperature to use, between `0` and `2`. Higher values like `0.8` will make the output more random, while lower values like `0.2` will make it more focused and deterministic.
- `presence_penalty`: Number between `-2.0` and `2.0`. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
- `frequency_penalty`: Number between `-2.0` and `2.0`. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.

### Methods

```python
__call__(
  memory: Memory,
  description: str = "You are a bot."
) -> Response
```

Process a conversation using the `OpenAI` model and return a [`Response`]() object.

This function sends a conversation stored in the `memory` parameter to the specified OpenAI model
(`self.model_name`), retrieves a response from the model, and records the conversation in memory.
It then constructs a [`Response`]() object containing the model's reply.

#### Args

- `memory` ([`Memory`]()): An instance of the [`Memory`]() class containing the conversation history.
- `description` (`str`, optional): Description of the model. Defaults to `"You are a bot."`.

#### Returns

- [`Response`](): A [`Response`]() object containing the model's reply, timestamp, latency, and model name.

#### Raises
- [`TokensExceeded`](): When the model exceeds the maximum number of tokens allowed.

```python
parse_filters(
  prompt: str,
  context: List[Message],
  filters: List[Filter]
) -> List[Message]
```

Generates a prompt message to check if a given prompt follows a set of filtering rules.

#### Args

- `prompt` (`str`): a string representing the prompt that will be checked against rules.
- `context` (`List[`[`Message`]()`]`): A list containing the last 3 messages from the chat.
- `filters` (`List[`[`Filter`]()`]`): List of filters used by the chatbot.

#### Returns

`List[`[`Message`]()`]`: a list of messages to be used by the chatbot to check if the prompt respects the rules

## Softtek OpenAI

```python
SofttekOpenAI(
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
)
```

Creates a Softtek OpenAI language model. This class is a subclass of the [`LLMModel`](#llm-model) abstract base class.

#### Args

- `api_key` (`str`): **LLMOPs** API key.
- `model_name` (`str`): Name of the model.
- `max_tokens` (`int | None`, optional): The maximum number of tokens to generate in the chat completion. The total length of input tokens and generated tokens is **limited by the model's context length**. Defaults to `None`.
- `temperature` (`float`, optional): What sampling temperature to use, between `0` and `2`. Higher values like `0.8` will make the output more random, while lower values like `0.2` will make it more focused and deterministic. Defaults to `1`.
- `presence_penalty` (`float`, optional): Number between `-2.0` and `2.0`. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics. Defaults to `0`.
- `frequency_penalty` (`float`, optional): Number between `-2.0` and `2.0`. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim. Defaults to `0`.
- `logit_bias` (`Dict[int, int] | None`, optional): A map of tokens to their desired logit bias. The keys are tokens, and the values are the bias to add to the logits (before applying temperature). Defaults to `None`.
- `stop` (`str | List[str] | None`, optional): One or more sequences where the API will stop generating further tokens. Defaults to `None`.
- `top_p` (`float`, optional): An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with `top_p` probability mass. So `0.1` means only the tokens comprising the top 10% probability mass are considered. Defaults to `1`.
- `user` (`str | None`, optional): The name of the user for which the API will generate a response. Defaults to `None`.
- `verbose` (`bool`, optional): Whether to print debug messages. Defaults to `False`.

### Properties

- `model_name`: The name of the model.
- `verbose`: Whether to print debug messages.
- `max_tokens`: The maximum number of tokens to generate in the chat completion.
- `temperature`: What sampling temperature to use, between `0` and `2`. Higher values like `0.8` will make the output more random, while lower values like `0.2` will make it more focused and deterministic.
- `presence_penalty`: Number between `-2.0` and `2.0`. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.
- `frequency_penalty`: Number between `-2.0` and `2.0`. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.
- `logit_bias`: A map of tokens to their desired logit bias. The keys are tokens, and the values are the bias to add to the logits (before applying temperature).
- `stop`: One or more sequences where the API will stop generating further tokens.
- `top_p`: An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with `top_p` probability mass. So `0.1` means only the tokens comprising the top 10% probability mass are considered.
- `user`: The name of the user for which the API will generate a response.
- `api_key`: The LMMOPs API key.

### Methods

```python
__call__(
  memory: Memory,
  description: str = "You are a bot.",
  logging_kwargs: Dict = {},
) -> Response
```

Conversational interface that interacts with the `SofttekOpenAI` Chat API to generate a response.

#### Args

- `memory` ([`Memory`]()): An instance of the [`Memory`]() class that holds previous conversation messages.
- `description` (`str`, optional): A description of the bot. Defaults to `"You are a bot."`.
- `logging_kwargs` (`Dict`, optional): A dictionary containing the parameters to be logged. Defaults to `{}`.

#### Raises

- `Exception`: If the API request to the Softtek OpenAI Chat API returns a status code other than `200`.

#### Returns

- [`Response`](): An instance of the [`Response`]() class that contains the generated response.

```python
parse_filters(
  prompt: str,
  context: List[Message],
  filters: List[Filter]
) -> List[Message]
```

Generates a prompt message to check if a given prompt follows a set of filtering rules.

#### Args

- `prompt` (`str`): a string representing the prompt that will be checked against rules.
- `context` (`List[`[`Message`]()`]`): A list containing the last 3 messages from the chat.
- `filters` (`List[`[`Filter`]()`]`): List of filters used by the chatbot.

#### Returns

- `List[`[`Message`]()`]`: a list of messages to be used by the chatbot to check if the prompt respects the rules.

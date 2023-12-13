# Embeddings

This module contains classes for embeddings models.

### Index

- [Embeddings Model](#embeddings-model)
- [OpenAI Embeddings](#openai-embeddings)
- [Softtek OpenAI Embeddings](#softtek-openai-embeddings)

## Embeddings Model

```python
EmbeddingsModel(
  **kwargs: Any
)
```

Creates an abstract base class for an embeddings model. Used as a base class for implementing different types of embeddings models.

### Methods

```python
embed(
  prompt: str,
  **kwargs: Any
) -> List[float]
```

This is an abstract method for embedding a prompt into a list of floats. **This method must be implemented by a subclass**.

#### Args

- `prompt` (`str`): The string prompt to embed.
- `**kwargs` (`Any`): Additional arguments for implementation-defined use.

#### Returns

- `List[float]`: The embedding of the prompt as a list of floats.

#### Raises

- `NotImplementedError`: When this abstract method is called without being implemented in a subclass.

## OpenAI Embeddings

```python
OpenAIEmbeddings(
  api_key: str,
  model_name: str,
  api_type: Literal["azure"] | None = None,
  api_base: str | None = None,
  api_version: str = "2023-07-01-preview",
)
```

Creates an OpenAI embeddings model. This class is a subclass of the [`EmbeddingsModel`](#embeddings-model) abstract base class.

#### Args

- `api_key` (`str`): OpenAI API key.
- `model_name` (`str`): OpenAI embeddings model name.
- `api_type` (`Literal["azure"] | None`, optional): Type of API to use. Defaults to `None`.
- `api_base` (`str | None`, optional): Base URL for Azure API. Defaults to `None`.
- `api_version` (`str`, optional): API version for Azure API. Defaults to `"2023-07-01-preview"`.

#### Raises

- `ValueError`: When `api_type` is not `"azure"` or `None`.

### Properties

- `model_name`: Embeddings model name.

### Methods

```python
embed(
  prompt: str,
  **kwargs
) -> List[float]
```

Embeds a prompt into a list of floats.

#### Args

- `prompt` (`str`): Prompt to embed.

#### Returns

- `List[float]`: Embedding of the prompt as a list of floats.

## Softtek OpenAI Embeddings

```python
SofttekOpenAIEmbeddings(
  model_name: str,
  api_key: str
)
```

Creates a Softtek OpenAI embeddings model. This class is a subclass of the [`EmbeddingsModel`](#embeddings-model) abstract base class.

#### Args

- `model_name` (`str`): Name of the embeddings model.
- `api_key` (`str`): API key for the Softtek OpenAI API.

### Properties

- `model_name`: Embeddings model name.

### Methods

```python
embed(
  prompt: str,
  additional_kwargs: Dict = {}
) -> List[float]
```

Embeds a prompt into a list of floats.

#### Args

- `prompt` (`str`): Prompt to embed.
- `additional_kwargs` (`Dict`, optional): Additional keyword arguments. Defaults to `{}`.

#### Returns

- `List[float]`: Embedding of the prompt as a list of floats.

#### Raises

- `Exception`: When the API returns a non-`200` status code.

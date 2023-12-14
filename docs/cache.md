# Cache

Represents the cache of the assistant. Stores all the prompts and responses that have been exchanged between the user and the assistant.

### Index

- [Cache](#cache-1)

## Cache

```python
Cache(
  vector_store: VectorStore,
  embeddings_model: EmbeddingsModel
)
```

Represents the cache of the assistant. Stores all the prompts and responses that have been exchanged between the user and the assistant.

#### Args

- `vector_store` ([`VectorStore`](vectorStores.md)): The vector store used to store the prompts and responses.
- `embeddings_model` ([`EmbeddingsModel`](embeddings.md)): The embeddings model used to generate embeddings for prompts.

### Properties

- `vector_store`: The [`vector store`](vectorStores.md) used to store the prompts and responses.
- `embeddings_model`: The [`embeddings model`](embeddings.md) used to generate embeddings for prompts.

### Methods

```python
add(
  prompt: str,
  response: Response,
  **kwargs
)
```

This function adds a prompt and response to the cache. It calculates the [`embeddings`](embeddings.md) for the prompt and adds it to the [`vector store`](vectorStores.md).

#### Args

- `prompt` (`str`): The prompt that was sent to the assistant.
- `response` ([`Response`](./schemas/response.md)): A [`Response`](./schemas/response.md) object containing the model's reply, timestamp, latency, and model name.

```python
retrieve(
  prompt: str,
  threshold: float = 0.9,
  additional_kwargs: Dict = {},
  **kwargs
) -> Tuple[Response | None, float]
```

This function retrieves the best [`response`](./schemas/response.md) from a query using the `prompt` provided by the user. It calculates the time taken to retrieve the data and returns the response.

#### Args

- `prompt` (`str`): The prompt that was sent to the assistant.
- `threshold` (`float`, optional): The threshold to use for the search. Defaults to `0.9`.
- `additional_kwargs` (`Dict`, optional): Optional dictionary of additional keyword arguments to add to the retrieved response. Defaults to `{}`.

#### Returns

- `Tuple[`[`Response`](./schemas/response.md)` | None, float]`: A tuple containing the [`response`](./schemas/response.md) and the **score** of the best match.

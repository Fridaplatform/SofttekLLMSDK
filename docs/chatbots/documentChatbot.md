# Document Chatbot

A chatbot that uses a knowledge base to answer questions. The knowledge base is a vector store that contains the documents. The embeddings model is used to embed the documents and the prompt. The model is used to generate the response.

### Index

- [Document Chatbot](#document-chatbot-1)

## Document ChatBot

```python
DocumentChatBot(
  model: LLMModel,
  knowledge_base: VectorStore,
  embeddings_model: EmbeddingsModel,
  description: str = "You are a helpful research assistant. You have access to documents and always respond using the most relevant information.",
  memory: Memory = WindowMemory(window_size=10),
  non_valid_response: str | None = None,
  filters: List[Filter] | None = None,
  cache: Cache | None = None,
  cache_probability: float = 0.5,
  verbose: bool = False,
  knowledge_base_namespace: str | None = None,
)
```

A chatbot that uses a knowledge base to answer questions. The knowledge base is a [`vector store`](../vectorStores.md) that contains the documents. The [`embeddings model`](../embeddings.md) is used to embed the documents and the prompt. The [`model`](../models.md) is used to generate the response. Inherits from [`Chatbot`](chatbot.md).

#### Args

- `model` ([`LLMModel`](../models.md)): The LLM to use for generating the [`response`](../schemas/response.md).
- `knowledge_base` ([`VectorStore`](../vectorStores.md)): The vector store that contains the documents.
- `embeddings_model` ([`EmbeddingsModel`](../embeddings.md)): The embeddings model to use for embedding the documents and the prompt.
- `description` (`str`, optional): Information about the bot. Defaults to `"You are a helpful research assistant. You have acess to documents and always respond using the most relevant information."`.
- `memory` ([`Memory`](../memory.md), optional): The memory to use. Defaults to [`WindowMemory`](../memory.md)`(window_size=10)`.
- `non_valid_response` (`str | None`, optional): [`Response`](../schemas/response.md) given when the prompt does not follow the rules set by the [`filters`](../schemas/filter.md). Defaults to `None`. If `None`, an [`InvalidPrompt`](../exceptions.md) exception is raised when the prompt does not follow the rules set by the `filters`.
- `filters` (`List[`[`Filter`](../schemas/filter.md)`] | None`, optional): List of filters used by the chatbot. Defaults to `None`.
- `cache` ([`Cache`](../cache.md)` | None`, optional): Cache used by the chatbot. Defaults to `None`.
- `cache_probability` (`float`, optional): Probability of using the [`cache`](../cache.md). Defaults to `0.5`. If `1.0`, the cache is **always used**. If `0.0`, the cache is **never used**.
- `verbose` (`bool`, optional): Whether to print additional information. Defaults to `False`.
- `knowledge_base_namespace` (`str | None`, optional): Namespace used by the knowledge base in the [`vector store`](../vectorStores.md). Defaults to `None`.

### Properties

- `model`: The [model](../models.md) used by the chatbot.
- `memory`: The [memory](../memory.md) used by the chatbot.
- `description`: Information about the chatbot.
- `filters`: The [filters](../schemas/filter.md) used by the chatbot.
- `cache`: The [cache](../cache.md) used by the chatbot.
- `cache_probability`: The probability of using the [cache](../cache.md).
- `verbose`: Whether to print additional information.
- `knowledge_base`: The [vector store](../vectorStores.md) that contains the documents.
- `embeddings_model`: The [embeddings model](../embeddings.md) to use for embedding the documents and the prompt.
- `knowledge_base_namespace`: Namespace used by the knowledge base in the [vector store](../vectorStores.md).

### Methods

```python
chat(
  prompt: str,
  print_cache_score: bool = False,
  include_context: bool = False,
  top_documents: int = 5,
  cache_kwargs: Dict = {},
  logging_kwargs: Dict | None = None,
) -> Response
```

Chatbot function that returns a [response](../schemas/response.md) given a `prompt`. If a [memory](../memory.md) and/or [cache](../cache.md) are available, it considers **previously stored conversations**. [`Filters`](../schemas/filter.md) are applied to the `prompt` before processing to ensure it is valid.

#### Args

- `prompt` (`str`): User's input string text.
- `print_cache_score` (`bool`, optional): Whether to print the [cache](../cache.md) score. Defaults to `False`.
- `include_context` (`bool`, optional): Whether to include the context in the [response](../schemas/response.md). Defaults to `False`.
- `top_documents` (`int`, optional): The number of documents to consider. Defaults to `5`.
- `cache_kwargs` (`Dict`, optional): Additional keyword arguments to be passed to the [cache](../cache.md). Defaults to `{}`.
- `logging_kwargs` (`Dict`, optional): additional keyword arguments to be passed to the logging function. **Can only be used with certain [models](../models.md)**. Defaults to `None`.

#### Raises

- [`InvalidPrompt`](../exceptions.md): If the `prompt` does not follow the rules set by the [filters](../schemas/filter.md) and **`non_valid_response` is `None`**.

#### Returns

- [`Response`](../schemas/response.md): The response given by the chatbot. Whithin the `additional_kwargs`, the following keys are available: `sources` (always), `context` (if `include_context` is `True`).

```python
add_document(
  file: str | bytes,
  file_type: Literal["pdf", "doc", "docx", "txt", "csv"],
  document_name: str | None = None,
)
```

Adds a document to the knowledge base.

#### Args

- `file` (`str | bytes`): Either the path to the file or the bytes of the file.
- `file_type` (`Literal["pdf", "doc", "docx", "txt", "csv"]`): The type of the file.
- `document_name` (`str | None`, optional): The name of the document. Defaults to `None`. If `None`, the name of the file is used.

```python
delete_document(
  file: str | bytes,
  file_type: Literal["pdf", "doc", "docx", "txt", "csv"],
  document_name: str | None = None,
)
```

Deletes a document from the knowledge base.

#### Args

- `file` (`str | bytes`): Either the path to the file or the bytes of the file.
- `file_type` (`Literal["pdf", "doc", "docx", "txt", "csv"]`): The type of the file.
- `document_name` (`str | None`, optional): The name of the document. Defaults to `None`. If `None`, the name of the file is used.

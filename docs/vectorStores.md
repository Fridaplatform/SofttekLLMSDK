# Vector Stores

Classes for managing vectors in a vector store.

### Index

- [Vector Store](#vector-store)
- [Pinecone Vector Store](#pinecone-vector-store)
- [FAISS Vector Store](#faiss-vector-store)
- [Softtek Vector Store](#softtek-vector-store)
- [Supabase Vector Store](#supabase-vector-store)

## Vector Store

```python
VectorStore()
```

Abstract class for managing vectors in a vector store.

### Methods

```python
add(
  vectors: List[Vector],
  **kwargs: Any
)
```

Abstract method for adding the given vectors to the vectorstore.

#### Args

- `vectors` (`List[`[`Vector`](./schemas/vector.md)`]`): A List of [`Vector`](./schemas/vector.md) instances to add.
- `**kwargs` (`Any`): Additional arguments.

#### Raises

- `NotImplementedError`: The method must be implemented by a subclass.

```python
delete(
  ids: List[str],
  **kwargs: Any
)
```

Abstract method for deleting vectors from the `VectorStore` given a list of vector IDs

#### Args

- `ids` (`List[str]`): A List of [`Vector`](./schemas/vector.md) IDs to delete.
- `**kwargs` (`Any`): Additional arguments.

#### Raises

- `NotImplementedError`: The method must be implemented by a subclass.

```python
search(
  vector: Vector | None = None,
  top_k: int = 1,
  **kwargs: Any
) -> List[Vector]
```

Abstract method for searching vectors that match the specified criteria.

#### Args

- `vector` ([`Vector`](./schemas/vector.md)` | None`, optional): The vector to use as a reference for the search. Defaults to `None`.
- `top_k` (`int`, optional): The number of results to return for each query. Defaults to `1`.
- `**kwargs` (`Any`): Additional keyword arguments to customize the search criteria.

#### Raises

- `NotImplementedError`: If the search method is not overridden.

## Pinecone Vector Store

```python
PineconeVectorStore(
  api_key: str,
  environment: str,
  index_name: str,
  proxy: str | None = None
)
```

Class for managing vectors in a Pinecone index. Inherits from [`VectorStore`](#vector-store).

#### Args

- `api_key` (`str`): The API key for authentication with the Pinecone service.
- `environment` (`str`): The Pinecone environment to use (e.g., `"production"` or `"sandbox"`).
- `index_name` (`str`): The name of the index where vectors will be stored and retrieved.
- `proxy` (`str | None`, optional): The proxy URL to use for requests. Defaults to `None`.

#### Note

- Make sure to use a valid API key and specify the desired environment and index name.

### Methods

```python
add(
  vectors: List[Vector],
  namespace: str | None = None,
  batch_size: int | None = None,
  show_progress: bool = True,
  **kwargs: Any,
)
```

Add vectors to the index.

Args:

- `vectors` (`List[`[`Vector`](./schemas/vector.md)`]`): A list of [`Vector`](./schemas/vector.md) objects to add to the index. Note that each vector must have a unique ID.
- `namespace` (`str | None`, optional): The namespace to write to. If not specified, the **default namespace** is used. Defaults to `None`.
- `batch_size` (`int | None`, optional): The number of vectors to upsert in each batch. If not specified, all vectors will be upserted in a single batch. Defaults to `None`.
- `show_progress` (`bool`, optional): Whether to show a progress bar using `tqdm`. Applied only if `batch_size` is provided. Defaults to `True`.
- `**kwargs` (`Any`): Additional arguments.

#### Raises

- `ValueError`: If any of the vectors do not have a unique ID.

```python
delete(
  ids: List[str] | None = None,
  delete_all: bool | None = None,
  namespace: str | None = None,
  filter: Dict | None = None,
  **kwargs: Any,
)
```

Delete vectors from the index.

#### Args

- `ids` (`List[str] | None`, optional): A list of vector IDs to delete. Defaults to `None`.
- `delete_all` (`bool | None`, optional): This indicates that **all vectors in the index namespace** should be deleted. Defaults to `None`.
- `namespace` (`str | None`, optional): The namespace to delete vectors from. If not specified, **the default namespace is used**. Defaults to `None`.
- `filter` (`Dict | None`, optional): If specified, the metadata filter here will be used to select the vectors to delete. This is mutually exclusive with specifying ids to delete in the `ids` param or using `delete_all=True`. Defaults to `None`.
- `**kwargs` (`Any`): Additional arguments.

```python
search(
  vector: Vector | None = None,
  id: str | None = None,
  top_k: int = 1,
  namespace: str | None = None,
  filter: Dict | None = None,
  **kwargs: Any,
) -> List[Vector]
```

Search for vectors in the index.

#### Args

- `vector` ([`Vector`](./schemas/vector.md)` | None`, optional): The query vector. Each call can contain only one of the parameters `id` or `vector`. Defaults to `None`.
- `id` (`str | None`, optional): The unique ID of the vector to be used as a query vector. Each call can contain only one of the parameters `id` or `vector`. Defaults to `None`.
- `top_k` (`int`, optional): The number of results to return for each query. Defaults to `1`.
- `namespace` (`str | None`, optional): The namespace to fetch vectors from. If not specified, **the default namespace is used**. Defaults to `None`.
- `filter` (`Dict | None`, optional): The filter to apply. You can use vector metadata to limit your search. Defaults to `None`.
- `**kwargs` (`Any`): Additional arguments.

#### Returns

- `List[`[`Vector`](./schemas/vector.md)`]`: A list of [`Vector`](./schemas/vector.md) objects containing the search results.

## FAISS Vector Store

```python
FAISSVectorStore(
  local_id: Dict[str | None, List[Vector]] = None,
  index: Dict[str | None, Any] = None,
  d: int = 1536,
)
```

Class for managing vectors in a FAISS index. Inherits from [`VectorStore`](#vector-store).

#### Args

- `local_id` (`Dict[str | None, List[`[`Vector`](./schemas/vector.md)`]]`, optional): A dictionary with the list of [`Vector`](./schemas/vector.md) objects of each namespace.
- `index` (`Dict[str | None, Any]`, optional): A dictionary with the FAISS index of each namespace.
- `d` (`int`, optional): The dimension of the Vector embeddings to be stored. Must coincide with the [embeddings model](./embeddings.md) used. The default is `1536`.

#### Raises

- `ValueError`: If the user provides only one of the arguments.

#### Note

- The `None` key in both arguments refers to the **general namespace**.

### Properties

- `local_id`: A dictionary with the list of [`Vector`](./schemas/vector.md) objects of each namespace.
- `index`: A dictionary with the index of each namespace.

### Methods

```python
add(
  vectors: List[Vector],
  namespace: str | None = None,
)
```

Adds the given [`Vector`](./schemas/vector.md) objects to the namespace.

If the namespace does not exist, it is created with the given method. If no method is provided, `IndexFlatIP` is used.

#### Args

- `vectors` (`List[`[`Vector`](./schemas/vector.md)`]`): The list of [`Vector`](./schemas/vector.md) objects to be added.
- `namespace` (`str | None`, optional): The namespace where the [`Vector`](./schemas/vector.md) objects are going to be added. The default is `None`.

#### Raises

- `ValueError`: if an id is not unique within the given vectors or within the namespace.
- `ValueError`: if the dimension (`d`) of any of the vectors is different to the dimension set in the index.

```python
delete(
  ids: List[str] | None = None,
  delete_all: bool = False,
  namespace: str | None = None,
)
```

Deletes the given [`Vector`](./schemas/vector.md) objects or all the Vector objects of the given namespace.

#### Args

- `ids` (`List[str] | None`, optional): The list of [`Vector`](./schemas/vector.md) objects to be deleted from the given namespace. The default is `None`.
- `delete_all` (`bool`, optional): If set to `True`, all the [`Vector`](./schemas/vector.md) objects will be deleted from the given namespace. The default is `False`.
- `namespace` (`str | None`, optional): The namespace where the [`Vector`](./schemas/vector.md) objects are going to be deleted from. The default is `None`.

#### Raises

- `ValueError`: if the namespace does not exist.
- `ValueError`: if neither `ids` nor `delete_all` are given.

#### Note

- You must provide either `ids` or `delete_all`. And if both are given **`ids` has the priority**.

```python
search(
  vector: Vector | None = None,
  id: str | None = None,
  top_k: int = 1,
  namespace: str | None = None,
  **kwargs,
) -> List[Vector]
```

Searches for the `top_k` closest [`Vector`](./schemas/vector.md) objects to the given Vector object or `id`.

#### Args

- `vector` ([`Vector`](./schemas/vector.md)` | None`, optional): The [`Vector`](./schemas/vector.md) object to be compared to. The default is `None`.
- `id` (`str | None`, optional): The id of the [`Vector`](./schemas/vector.md) object to be compared to. The default is `None`.
- `top_k` (`int`, optional): The number of top [`Vector`](./schemas/vector.md) objects to be returned. The default is `1`.
- `namespace` (`str | None`, optional): The namespace of the index that is going to be used. The default is `None`.

#### Returns

- `List[`[`Vector`](./schemas/vector.md)`]`: The list of top [`Vector`](./schemas/vector.md) objects.

#### Raises

- `ValueError`: if the namespace does not exist.
- `ValueError`: if neither `vector` nor `id` are given.

#### Note

- You must provide either `vector` or `id`. If both are given **`vector` has the priority**.

```python
save_local(
  dir_path: str = ".",
  namespace: str | None = None,
  save_all: bool = False,
)
```

Saves both the index and the `local_id` objects from the given namespace or from all the namespaces.

If `folder_path` is not provided, it is stored in the **current directory**.
If the folder does not exist, **it is created**.

#### Args

- `dir_path` (`str`, optional): The path to which all the files will be saved. The default is the current directory.
- `namespace` (`str | None`, optional): The namespace that will be saved. The default is `None`.
- `save_all` (`bool`, optional): If set to `True`, all the namespaces will be saved. The default is `False`.

#### Raises

- `ValueError`: if the namespace does not exist.

#### Note

- You must provide either `namespace` or `save_all`. If both are given `save_all` has the priority.

```python
@classmethod
FAISSVectorStore.load_local(
  namespaces: List[str | None],
  dir_path: str = ".",
  d: int = 1536,
)
```

Creates a `FAISSVectorStore` from a list of `namespaces` stored in the `dir_path`.

#### Args

- `namespaces` (`List[str | None]`): The namespaces that will be retrieved.
- `dir_path` (`str`, optional): The path to which all the files will be retrieved. The default is the current directory.
- `d` (`int`, optional): The dimension of the [`Vector`](./schemas/vector.md) embeddings to be stored. Must coincide with the [embeddings model](./embeddings.md) used. The default is `1536`.

#### Raises

- `ValueError`: if the given directory does not exist.
- `ValueError`: if something goes wrong with the files.

#### Note

- If you want to load the default index, include `None` in the list.
- Only if both the `.faiss` and `.pkl` files are found, the namespace is loaded.
- If a namespace raises an error, it will be passed.

## Softtek Vector Store

```python
SofttekVectorStore(
  api_key: str
)
```

Class for managing vectors in a Softtek index. Inherits from [`VectorStore`](#vector-store).

#### Args

- `api_key` (`str`): The API key for authentication with the **LLMOPs service**.

### Properties

- `api_key`: The API key for authentication with the **LLMOPs service**.

### Methods

```python
add(
  vectors: List[Vector],
  namespace: str | None = None,
  **kwargs: Any,
)
```

Add vectors to the index.

#### Args

- `vectors` (`List[`[`Vector`](./schemas/vector.md)`]`): A list of [`Vector`](./schemas/vector.md) objects to add to the index. Note that **each vector must have a unique ID**.
- `namespace` (`str | None`, optional): The namespace to write to. If not specified, **the default namespace is used**. Defaults to `None`.

#### Raises

- `ValueError`: If any of the vectors do not have a unique ID.
- `ValueError`: If any of the vectors do not have embeddings.
- `Exception`: If the request fails.

```python
delete(
  ids: List[str] | None = None,
  delete_all: bool | None = None,
  namespace: str | None = None,
  filter: Dict | None = None,
  **kwargs: Any,
)
```

Delete vectors from the index.

#### Args

- `ids` (`List[str] | None`, optional): A list of vector IDs to delete. Defaults to `None`.
- `delete_all` (`bool | None`, optional): This indicates that **all vectors in the index namespace** should be deleted. Defaults to `None`.
- `namespace` (`str | None`, optional): The namespace to delete vectors from. If not specified, **the default namespace is used**. Defaults to `None`.
- `filter` (`Dict | None`, optional): If specified, the metadata filter here will be used to select the vectors to delete. This is mutually exclusive with specifying ids to delete in the `ids` param or using `delete_all=True`. Defaults to `None`.

#### Raises

- `Exception`: If the request fails.

```python
search(
  vector: Vector | None = None,
  id: str | None = None,
  top_k: int = 1,
  namespace: str | None = None,
  filter: Dict | None = None,
  **kwargs: Any,
) -> List[Vector]
```

Search for vectors in the index.

#### Args

- `vector` ([`Vector`](./schemas/vector.md)` | None`, optional): The query vector. Each call can contain only one of the parameters `id` or `vector`. Defaults to `None`.
- `id` (`str | None`, optional): The unique ID of the vector to be used as a query vector. Each call can contain only one of the parameters `id` or `vector`. Defaults to `None`.
- `top_k` (`int`, optional): The number of results to return for each query. Defaults to `1`.
- `namespace` (`str | None`, optional): The namespace to fetch vectors from. If not specified, **the default namespace is used**. Defaults to `None`.
- `filter` (`Dict | None`, optional): The filter to apply. You can use vector metadata to limit your search. Defaults to `None`.

#### Raises

- `Exception`: If the request fails.

#### Returns

- `List[`[`Vector`](./schemas/vector.md)`]`: A list of [`Vector`](./schemas/vector.md) objects containing the search results.

## Supabase Vector Store

```python
SupabaseVectorStore(
  api_key: str,
  url: str,
  index_name: str
)
```

Class for managing vectors in a Supabase table.

Initialize a SupabaseVectorStore object for managing vectors in a Supabase table. Inherits from [`VectorStore`](#vector-store).

#### Args

- `api_key` (`str`): The API key for authentication with the Supabase service.
- `url` (`str`): The Supabase URL.
- `index_name` (`str`): The name of the table where vectors will be stored and retrieved.

### Methods

```python
add(
  vectors: List[Vector],
  **kwargs: Any
)
```

Add vectors to the index.

#### Args

- `vectors` (`List[`[`Vector`](./schemas/vector.md)`]`): A list of [`Vector`](./schemas/vector.md) objects to add to the index. **Note that each vector must have a unique ID**.

#### Raises

- `ValueError`: If any of the vectors do not have embeddings.

#### Note

- Requires a table with columns: `id` (text), `vector` (vector(1536 or dimension of [embeddings model](./embeddings.md) used)), `metadata` (json), `created_at` (timestamp).
- **Vector type is enabled with the vector extension for postgres in supabase**.
- Requires default value of `id` to `gen_random_uuid()`.

```python
delete(
  ids: List[str] | None = None,
  **kwargs: Any
)
```

Delete vectors from the index.

#### Args

- `ids` (`List[str] | None`, optional): A list of vector IDs to delete. Defaults to `None`.

```python
search(
  vector: Vector | None = None,
  top_k: int = 1,
  **kwargs: Any
) -> List[Vector]
```

Search for vectors in the index.

#### Args

- `vector` ([`Vector`](./schemas/vector.md)` | None`, optional): The query [`vector`](./schemas/vector.md). Defaults to `None`.
- `top_k` (`int`, optional): the number of vectors to retrieve

#### Returns

- `List[`[`Vector`](./schemas/vector.md)`]`: A list of [`Vector`](./schemas/vector.md) objects containing the search results.

#### Note

- Requires the following procedure (where you only change the value of the `TABLENAME` variable):

```sql
drop function if exists similarity_search_TABLENAME (embedding vector (1536), match_count bigint);

create or replace function similarity_search_TABLENAME(embedding vector(1536), match_count bigint)
returns table (id text,similarity float, value vector(1536), ,metadata json)
language plpgsql
as $$
begin
return query
select
    TABLENAME.id,
    (TABLENAME.vector <#> embedding) * -1 as similarity,
    TABLENAME.vector,
    TABLENAME.metadata
from TABLENAME
order by TABLENAME.vector <#> embedding
limit match_count;
end;
$$;
```

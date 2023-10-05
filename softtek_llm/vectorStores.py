from abc import ABC, abstractmethod
from typing import Any, Dict, List
from pinecone.core.client.configuration import Configuration as OpenApiConfiguration

import pinecone
from typing_extensions import override

from softtek_llm.schemas import Vector
from supabase import Client, create_client


class VectorStore(ABC):
    def __init__(self):
        """Initializes the VectorStoreModel class."""
        super().__init__()

    @abstractmethod
    def add(self, vectors: List[Vector], **kwargs: Any):
        """
        Abstract method for adding the given vectors to the vectorstore.

        Args:
            `vectors` (List[Vector]): A List of Vector instances to add.
            `**kwargs` (Any): Additional arguments.

        Raises:
            NotImplementedError: The method must be implemented by a subclass.
        """
        raise NotImplementedError("add method must be overridden")

    @abstractmethod
    def delete(self, ids: List[str], **kwargs: Any):
        """
        Abstract method for deleting vectors from the VectorStore given a list of vector IDs

        Args:
            `ids` (List[str]): A List of Vector IDs to delete.
            `**kwargs` (Any): Additional arguments.

        Raises:
            NotImplementedError: The method must be implemented by a subclass.
        """
        raise NotImplementedError("delete method must be overridden")

    @abstractmethod
    def search(
        self, vector: Vector | None = None, top_k: int = 1, **kwargs: Any
    ) -> List[Vector]:
        """
        Abstract method for searching vectors that match the specified criteria.

        Args:
            `vector` (Vector, optional): The vector to use as a reference for the search. Defaults to None.
            `top_k` (int, optional): The number of results to return for each query. Defaults to 1.
            `**kwargs` (Any): Additional keyword arguments to customize the search criteria.

        Raises:
            NotImplementedError: If the search method is not overridden.
        """
        raise NotImplementedError("search method must be overridden")


class PineconeVectorStore(VectorStore):
    @override
    def __init__(
        self, api_key: str, environment: str, index_name: str, proxy: str | None = None
    ):
        """
        Initialize a PineconeVectorStore object for managing vectors in a Pinecone index.

        Args:
            `api_key` (str): The API key for authentication with the Pinecone service.
            `environment` (str): The Pinecone environment to use (e.g., "production" or "sandbox").
            `index_name` (str): The name of the index where vectors will be stored and retrieved.
            `proxy` (str | None, optional): The proxy URL to use for requests. Defaults to None.

        Note:
            Make sure to use a valid API key and specify the desired environment and index name.
        """
        if proxy is None:
            pinecone.init(api_key=api_key, environment=environment)
        else:
            openapi_config = OpenApiConfiguration.get_default_copy()
            openapi_config.proxy = proxy
            pinecone.init(
                api_key=api_key, environment=environment, openapi_config=openapi_config
            )
        self.__index = pinecone.Index(index_name)

    @override
    def add(
        self,
        vectors: List[Vector],
        namespace: str | None = None,
        batch_size: int | None = None,
        show_progress: bool = True,
        **kwargs: Any,
    ):
        """Add vectors to the index.

        Args:
            `vectors` (List[Vector]): A list of Vector objects to add to the index. Note that each vector must have a unique ID.
            `namespace` (str | None, optional): The namespace to write to. If not specified, the default namespace is used. Defaults to None.
            `batch_size` (int | None, optional): The number of vectors to upsert in each batch. If not specified, all vectors will be upserted in a single batch. Defaults to None.
            `show_progress` (bool, optional): Whether to show a progress bar using tqdm. Applied only if batch_size is provided. Defaults to True.
            `**kwargs` (Any): Additional arguments.

        Raises:
            ValueError: If any of the vectors do not have a unique ID.
        """
        data_to_add = []
        ids = []
        for vector in vectors:
            if not vector.id:
                raise ValueError("Vector ID cannot be empty when adding to Pinecone.")
            if vector.id in ids:
                raise ValueError(
                    f"Vector ID {vector.id} is not unique to this batch. Please make sure all vectors have unique IDs."
                )
            data_to_add.append((vector.id, vector.embeddings, vector.metadata))
            ids.append(vector.id)

        self.__index.upsert(
            data_to_add,
            namespace=namespace,
            batch_size=batch_size,
            show_progress=show_progress,
            **kwargs,
        )

    @override
    def delete(
        self,
        ids: List[str] | None = None,
        delete_all: bool | None = None,
        namespace: str | None = None,
        filter: Dict | None = None,
        **kwargs: Any,
    ):
        """Delete vectors from the index.

        Args:
            `ids` (List[str] | None, optional): A list of vector IDs to delete. Defaults to None.
            `delete_all` (bool | None, optional): This indicates that all vectors in the index namespace should be deleted. Defaults to None.
            `namespace` (str | None, optional): The namespace to delete vectors from. If not specified, the default namespace is used. Defaults to None.
            `filter` (Dict | None, optional): If specified, the metadata filter here will be used to select the vectors to delete. This is mutually exclusive with specifying ids to delete in the `ids` param or using `delete_all=True`. Defaults to None.
            `**kwargs` (Any): Additional arguments.
        """
        self.__index.delete(
            ids=ids, delete_all=delete_all, namespace=namespace, filter=filter, **kwargs
        )

    @override
    def search(
        self,
        vector: Vector | None = None,
        id: str | None = None,
        top_k: int = 1,
        namespace: str | None = None,
        filter: Dict | None = None,
        **kwargs: Any,
    ) -> List[Vector]:
        """Search for vectors in the index.

        Args:
            `vector` (Vector | None, optional): The query vector. Each call can contain only one of the parameters `id` or `vector`. Defaults to None.
            `id` (str | None, optional): The unique ID of the vector to be used as a query vector. Each call can contain only one of the parameters `id` or `vector`. Defaults to None.
            `top_k` (int, optional): The number of results to return for each query. Defaults to 1.
            `namespace` (str | None, optional): The namespace to fetch vectors from. If not specified, the default namespace is used. Defaults to None.
            `filter` (Dict | None, optional): The filter to apply. You can use vector metadata to limit your search. Defaults to None.
            `include_metadata` (bool, optional): Indicates whether metadata is included in the response as well as the ids. If omitted the server will use the default value of False. Defaults to None.
            `**kwargs` (Any): Additional arguments.

        Returns:
            `vectors` (List[Vector]): A list of Vector objects containing the search results.
        """
        # TODO: Default queries and sparse_vector parameters. Is QueryVector class iterable?
        query_response = self.__index.query(
            vector=vector.embeddings if vector else None,
            id=id,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_values=True,
            include_metadata=True,
            **kwargs,
        )

        vectors = []
        for match in query_response.matches:
            metadata = vector.metadata if vector else {}
            metadata.update(match.metadata)
            metadata.update({"score": match.score})
            vectors.append(
                Vector(
                    embeddings=match.values,
                    id=match.id,
                    metadata=metadata,
                )
            )

        return vectors


class SupabaseVectorStore(VectorStore):
    @override
    def __init__(self, api_key: str, url: str, index_name: str):
        """
        Initialize a SupabaseVectorStore object for managing vectors in a Supabase table.
        """
        self.__client = create_client(url, api_key)
        self.__index_name = index_name

    @override
    def add(self, vectors: List[Vector], **kwargs: Any):
        """
        Add vectors to the index.
        -- Requires a table with columns: id (text), vector (vector(1536)), metadata (json), created_at (timestamp)
        -- vector type is enabled with the vector extension for postgres in supabase
        -- requires default value of id to gen_random_uuid()

        """
        for vector in vectors:
            # if not vector.id:
            #     raise ValueError("Vector ID cannot be empty when adding to Supabase.")
            if not vector.embeddings:
                raise ValueError(
                    "Vector embeddings cannot be empty when adding to Supabase."
                )
            vec = {"vector": vector.embeddings, "metadata": vector.metadata}
            if vector.id is not None and vector.id != "":
                print("id is not none")
                vec["id"] = vector.id
            print(vec)
            self.__client.table(self.__index_name).insert(vec).execute()

    @override
    def delete(self, ids: List[str] | None = None, **kwargs: Any):
        """Delete vectors from the index.

        Args:
            ids (List[str] | None, optional): A list of vector IDs to delete. Defaults to None.
        """
        self.__client.table(self.__index_name).delete().in_("id", ids).execute()

    @override
    def search(
        self, vector: Vector | None = None, limit: int = 1, **kwargs: Any
    ) -> List[Vector]:
        """
        Search for vectors in the index.

        Args:
            vector (Vector | None, optional): The query vector. Each call can contain only one of the parameters `id` or `vector`. Defaults to None.
            limit (int, optional): the number of vectors to retrieve

        Returns:
            List[Vector]: A list of Vector objects containing the search results.

        -- Requires the following procedure (where you only change the value of the TABLENAME variable):
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
        """
        query_response = self.__client.rpc(
            "similarity_search_" + self.__index_name,
            {"embedding": vector.embeddings, "match_count": limit},
        ).execute()
        vectors = []
        print(query_response.data)
        for match in query_response.data:
            print(match)
            metadata = vector.metadata if vector else {}
            metadata.update(match["metadata"])
            metadata["score"] = match["similarity"]
            parsed_vector = [float(i) for i in match["value"][1:-1].split(",")]
            vectors.append(
                Vector(
                    embeddings=parsed_vector,
                    id=match["id"],
                    metadata=metadata,
                )
            )
        return vectors

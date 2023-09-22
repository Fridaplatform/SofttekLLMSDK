import unittest
from unittest.mock import patch, MagicMock
from softtek_llm.vectorStores import PineconeVectorStore, Vector

class TestPineconeVectorStoreInit(unittest.TestCase):

    @patch('pinecone.init')
    @patch('pinecone.Index')
    def test_init(self, mock_pinecone_index, mock_pinecone_init):
        # Mock the Pinecone init function
        api_key = "your_api_key"
        environment = "your_environment"
        index_name = "your_index_name"

        # Create an instance of PineconeVectorStore
        vector_store = PineconeVectorStore(api_key, environment, index_name)

        # Assert that pinecone.init was called with the correct arguments
        mock_pinecone_init.assert_called_once_with(api_key=api_key, environment=environment)

        # Assert that pinecone.Index was called with the correct index_name
        mock_pinecone_index.assert_called_once_with(index_name)

    @patch('pinecone.Index')
    def setUp(self, mock_pinecone_index):
        # Mock the Pinecone Index instance
        self.mock_index = MagicMock()
        mock_pinecone_index.return_value = self.mock_index

        # Create an instance of PineconeVectorStore
        self.api_key = "your_api_key"
        self.environment = "your_environment"
        self.index_name = "your_index_name"
        self.vector_store = PineconeVectorStore(self.api_key, self.environment, self.index_name)

    def test_add_vectors(self):
        # Create some sample Vector objects
        vectors = [
            Vector(id="1", embeddings=[0.1, 0.2, 0.3], metadata={"name": "vector_1"}),
            Vector(id="2", embeddings=[0.4, 0.5, 0.6], metadata={"name": "vector_2"}),
            Vector(id="3", embeddings=[0.7, 0.8, 0.9], metadata={"name": "vector_3"}),
        ]

        # Call the add method
        self.vector_store.add(vectors)

        # Assert that upsert was called with the correct data
        self.mock_index.upsert.assert_called_once_with(
            [
                ("1", [0.1, 0.2, 0.3], {"name": "vector_1"}),
                ("2", [0.4, 0.5, 0.6], {"name": "vector_2"}),
                ("3", [0.7, 0.8, 0.9], {"name": "vector_3"}),
            ],
            namespace=None,
            batch_size=None,
            show_progress=True,
        )

    def test_add_vectors_with_namespace(self):
        # Create some sample Vector objects
        vectors = [
            Vector(id="1", embeddings=[0.1, 0.2, 0.3], metadata={"name": "vector_1"}),
            Vector(id="2", embeddings=[0.4, 0.5, 0.6], metadata={"name": "vector_2"}),
        ]
        namespace = "custom_namespace"

        # Call the add method with a custom namespace
        self.vector_store.add(vectors, namespace=namespace)

        # Assert that upsert was called with the correct namespace
        self.mock_index.upsert.assert_called_once_with(
            [
                ("1", [0.1, 0.2, 0.3], {"name": "vector_1"}),
                ("2", [0.4, 0.5, 0.6], {"name": "vector_2"}),
            ],
            namespace=namespace,
            batch_size=None,
            show_progress=True,
        )

    def test_delete_ids(self):
        # Define a list of vector IDs to delete
        ids_to_delete = ["1", "2", "3"]

        # Create some sample Vector objects
        vectors = [
            Vector(id="1", embeddings=[0.1, 0.2, 0.3]),
            Vector(id="2", embeddings=[0.4, 0.5, 0.6]),
            Vector(id="3", embeddings=[0.7, 0.8, 0.9]),
        ]

        # Add the vectors to the index
        self.vector_store.add(vectors)

        # Call the delete method with a list of IDs
        self.vector_store.delete(ids=ids_to_delete)

        # Assert that delete was called with the correct IDs
        self.mock_index.delete.assert_called_once_with(
            ids=ids_to_delete,
            delete_all=None,
            namespace=None,
            filter=None,
        )

    def test_delete_all(self):
        # Create some sample Vector objects
        vectors = [
            Vector(id="1", embeddings=[0.1, 0.2, 0.3]),
            Vector(id="2", embeddings=[0.4, 0.5, 0.6]),
            Vector(id="3", embeddings=[0.7, 0.8, 0.9]),
        ]

        # Add the vectors to the index
        self.vector_store.add(vectors)

        # Call the delete method with delete_all=True
        self.vector_store.delete(delete_all=True)

        # Assert that delete was called with delete_all=True
        self.mock_index.delete.assert_called_once_with(
            ids=None,
            delete_all=True,
            namespace=None,
            filter=None,
        )

    def test_delete_with_namespace(self):
        # Create some sample Vector objects
        vectors = [
            Vector(id="1", embeddings=[0.1, 0.2, 0.3]),
            Vector(id="2", embeddings=[0.4, 0.5, 0.6]),
        ]

        # Add the vectors to the index
        self.vector_store.add(vectors)

        # Call the delete method with a custom namespace
        namespace = "custom_namespace"
        self.vector_store.delete(namespace=namespace)

        # Assert that delete was called with the correct namespace
        self.mock_index.delete.assert_called_once_with(
            ids=None,
            delete_all=None,
            namespace=namespace,
            filter=None,
        )

    def test_delete_with_filter(self):
        # Create some sample Vector objects
        vectors = [
            Vector(id="1", embeddings=[0.1, 0.2, 0.3], metadata={"field1": "value1"}),
            Vector(id="2", embeddings=[0.4, 0.5, 0.6], metadata={"field1": "value2"}),
            Vector(id="3", embeddings=[0.7, 0.8, 0.9], metadata={"field1": "value3"}),
        ]

        # Add the vectors to the index
        self.vector_store.add(vectors)

        # Define a filter dictionary
        filter_dict = {"field1": "value2"}

        # Call the delete method with a filter
        self.vector_store.delete(filter=filter_dict)

        # Assert that delete was called with the correct filter
        self.mock_index.delete.assert_called_once_with(
            ids=None,
            delete_all=None,
            namespace=None,
            filter=filter_dict,
        )

if __name__ == '__main__':
    unittest.main()

import unittest
import os
import shutil
from softtek_llm.vectorStores import FAISSVectorStore, Vector


class TestFAISSVectorStoreInit(unittest.TestCase):
    namespace = "test-1"
    path = "tempo_testing"
    vectors_1 = [
        Vector(id="1", embeddings=[0.1, 0.2, 0.3], metadata={"name": "vector_1"}),
        Vector(id="2", embeddings=[0.4, 0.5, 0.6], metadata={"name": "vector_2"}),
        Vector(id="3", embeddings=[0.7, 0.8, 0.9], metadata={"name": "vector_3"}),
    ]
    vectors_2 = [
        Vector(id="4", embeddings=[0.1, 0.2, 0.3], metadata={"name": "vector_4"}),
        Vector(id="5", embeddings=[0.4, 0.5, 0.6], metadata={"name": "vector_5"}),
    ]
    vector = Vector(id="6", embeddings=[0.3, 0.2, 0.1], metadata={"name": "vector_6"})

    def setUp(self):
        self.vector_store = FAISSVectorStore(d=3)

    def test_empty_vector_store(self):
        # Assert the index and local_id
        self.assertEqual(self.vector_store.index[None].ntotal, 0)
        self.assertEqual(self.vector_store.local_id[None], list())

    def test_add_vectors(self):
        # Adds vectors
        self.vector_store.add(vectors=self.vectors_1)

        # Assert the local_id vectors and the index size
        self.assertEqual(self.vector_store.local_id[None], self.vectors_1)
        self.assertEqual(self.vector_store.index[None].ntotal, 3)

    def test_add_vectors_with_namespace(self):
        # Add vectors
        self.vector_store.add(vectors=self.vectors_2, namespace=self.namespace)

        # Assert the local_id vectors and the index size
        self.assertEqual(self.vector_store.local_id[self.namespace], self.vectors_2)
        self.assertEqual(self.vector_store.index[self.namespace].ntotal, 2)

    def test_search_vector(self):
        # Adds vectors
        self.vector_store.add(vectors=self.vectors_1)
        self.vector_store.add(vectors=self.vectors_2)

        # Search
        vectors = self.vector_store.search(vector=self.vector, top_k=3)

        # Assert the top_k = 1 to the first vector of vector_1
        self.assertEqual(vectors, [self.vectors_1[2], self.vectors_2[1], self.vectors_1[1]])

    def test_save_local_and_load_local(self):
        # Adds vectors
        self.vector_store.add(vectors=self.vectors_1)
        self.vector_store.add(vectors=self.vectors_2, namespace=self.namespace)

        # Save the data
        self.vector_store.save_local(dir_path=self.path, save_all=True)

        # Assert the file exists (both .faiss and .pkl)
        self.assertTrue(os.path.isfile(os.path.join(self.path, "index.faiss")))
        self.assertTrue(os.path.isfile(os.path.join(self.path, "index.pkl")))
        self.assertTrue(
            os.path.isfile(os.path.join(self.path, f"{self.namespace}_index.faiss"))
        )
        self.assertTrue(
            os.path.isfile(os.path.join(self.path, f"{self.namespace}_index.pkl"))
        )

        # Create object from a list of namespaces
        vs = FAISSVectorStore.load_local(
            dir_path=self.path, namespaces=[None, self.namespace], d=3
        )

        # Asser the Class and the keys loaded
        self.assertTrue(isinstance(vs, FAISSVectorStore))
        self.assertEqual(list(vs.index.keys()), [None, self.namespace])
        self.assertEqual(list(vs.local_id.keys()), [None, self.namespace])

        # Delete the directory with the data
        if os.path.isdir(self.path):
            shutil.rmtree(self.path)

    def test_delete_vectors(self):
        # Add vectors
        self.vector_store.add(vectors=self.vectors_2, namespace=self.namespace)

        # Delete vector 1 from the namespace given
        self.vector_store.delete(ids=["4"], namespace=self.namespace)

        # Assert the local_id with the vector "5" and the index size
        self.assertEqual(self.vector_store.local_id[self.namespace][0].id, "5")
        self.assertEqual(self.vector_store.index[self.namespace].ntotal, 1)

    def test_delete_all(self):
        # Adds vectors
        self.vector_store.add(vectors=self.vectors_1)

        # Delete all in the None namespace
        self.vector_store.delete(delete_all=True)

        # Assert the local_id vectors and the index size
        self.assertEqual(self.vector_store.local_id[None], list())
        self.assertEqual(self.vector_store.index[None].ntotal, 0)


if __name__ == "__main__":
    unittest.main()

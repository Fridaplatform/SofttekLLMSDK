import unittest
from datetime import datetime
from time import perf_counter_ns
from unittest.mock import Mock, patch

import numpy as np

from softtek_llm.cache import Cache, EmbeddingsModel, Message, Response, Vector, VectorStore


class TestCacheClass(unittest.TestCase):
    def setUp(self):
        self.embeddings_model = Mock(spec=EmbeddingsModel)
        self.vector_store = Mock(spec=VectorStore)

        self.cache = Cache(self.vector_store, self.embeddings_model)

    def tearDown(self):
        self.cache = None
        self.vector_store = None
        self.embeddings_model = None

    @patch("softtek_llm.cache.uuid1")
    def test_add(self, mock_uuid1):
        prompt = "test prompt"
        response = Response(
            message=Message(role="assistant", content="test response"),
            created=int(datetime.utcnow().timestamp()),
            latency=0,
            from_cache=False,
            model="",
            additional_kwargs={},
        )
        embeddings = np.array([0.1, 0.2, 0.3])

        self.embeddings_model.embed.return_value = embeddings
        self.cache.add(prompt, response)

        self.embeddings_model.embed.assert_called_once_with(prompt)
        self.vector_store.add.assert_called_once_with(
            [
                Vector(
                    embeddings=embeddings,
                    id=str(mock_uuid1.return_value),
                    metadata={"response": "test response", "model": ""},
                )
            ]
        )

    # def test_retrieve_existing(self):
    #     prompt = "test prompt"
    #     embeddings = np.array([0.1, 0.2, 0.3])
    #     vector = Vector(
    #         embeddings=embeddings,
    #         metadata={"response": "test response", "model": "", "score": 1.0},
    #     )
    #     expected_response = Response(
    #         message=Message(role="assistant", content="test response"),
    #         created=int(datetime.utcnow().timestamp()),
    #         latency=0,
    #         from_cache=True,
    #         model="",
    #         additional_kwargs={},
    #     )

    #     self.embeddings_model.embed.return_value = embeddings
    #     self.vector_store.search.return_value = [vector]

    #     response = self.cache.retrieve(prompt)

    #     self.embeddings_model.embed.assert_called_once_with(prompt)
    #     self.vector_store.search.assert_called_once_with(
    #         Vector(embeddings=embeddings)
    #     )
    #     self.assertEqual(response, expected_response)

    def test_retrieve_not_found(self):
        prompt = "test prompt"
        embeddings = np.array([0.1, 0.2, 0.3])

        self.embeddings_model.embed.return_value = embeddings
        self.vector_store.search.return_value = []

        response = self.cache.retrieve(prompt)

        self.embeddings_model.embed.assert_called_once_with(prompt)
        self.vector_store.search.assert_called_once_with(
            Vector(embeddings=embeddings),
        )
        self.assertIsNone(response[0])

    def test_retrieve_low_score(self):
        prompt = "test prompt"
        embeddings = np.array([0.1, 0.2, 0.3])
        vector = Vector(
            embeddings=embeddings,
            metadata={"response": "test response", "model": "", "score": 0.8},
        )

        self.embeddings_model.embed.return_value = embeddings
        self.vector_store.search.return_value = [vector]

        response = self.cache.retrieve(prompt)

        self.embeddings_model.embed.assert_called_once_with(prompt)
        self.vector_store.search.assert_called_once_with(
            Vector(embeddings=embeddings),
        )
        self.assertIsNone(response[0])

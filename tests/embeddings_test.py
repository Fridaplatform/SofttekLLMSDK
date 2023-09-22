import os
import unittest

from dotenv import load_dotenv

from softtek_llm.embeddings import OpenAIEmbeddings

load_dotenv()


class TestOpenAIEmbeddings(unittest.TestCase):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable must be set.")
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base is None:
        raise ValueError("OPENAI_API_BASE environment variable must be set.")
    model_name = os.getenv("OPENAI_EMBEDDINGS_MODEL_NAME")
    if model_name is None:
        raise ValueError(
            "OPENAI_EMBEDDINGS_MODEL_NAME environment variable must be set."
        )

    def test_bad_api_type(self):
        with self.assertRaises(ValueError):
            OpenAIEmbeddings(
                api_key=self.api_key, model_name=self.model_name, api_type="notazure"
            )

    def test_azure_api(self):
        api_type = "azure"

        embeddings_model = OpenAIEmbeddings(
            api_key=self.api_key,
            model_name=self.model_name,
            api_type=api_type,
            api_base=self.api_base,
        )

        self.assertEqual(embeddings_model.model_name, self.model_name)

        embeddings = embeddings_model.embed("Hello, world!")
        self.assertIsInstance(embeddings, list)
        self.assertIsInstance(embeddings[0], float)
        self.assertEqual(len(embeddings), 1536)

    def test_azure_api_base_none(self):
        api_type = "azure"

        with self.assertRaises(ValueError):
            OpenAIEmbeddings(
                api_key=self.api_key,
                model_name=self.model_name,
                api_type=api_type,
            )

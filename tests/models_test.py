import os
import unittest

from openai.error import InvalidRequestError

from softtek_llm.memory import Memory
from softtek_llm.models import OpenAI
from softtek_llm.schemas import Response


class TestOpenAI(unittest.TestCase):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable must be set.")
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base is None:
        raise ValueError("OPENAI_API_BASE environment variable must be set.")
    model_name = os.getenv("OPENAI_CHAT_MODEL_NAME")
    if model_name is None:
        raise ValueError("OPENAI_CHAT_MODEL_NAME environment variable must be set.")

    def test_bad_api_type(self):
        with self.assertRaises(
            ValueError, msg="api_type must be either 'azure' or None, not notazure"
        ):
            OpenAI(
                api_key=self.api_key, model_name=self.model_name, api_type="notazure"
            )

    def test_azure_api_base_none(self):
        api_type = "azure"

        with self.assertRaises(ValueError, msg="api_base must be set for Azure API"):
            OpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                api_type=api_type,
            )

    def test_bad_max_tokens(self):
        api_type = "azure"

        with self.assertRaises(TypeError, msg="max_tokens must be an integer or None"):
            OpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                api_type=api_type,
                max_tokens="notint",
            )

    def test_good_max_tokens(self):
        api_type = "azure"

        openai = OpenAI(
            api_key=self.api_key,
            model_name=self.model_name,
            api_type=api_type,
            api_base=self.api_base,
            max_tokens=10,
        )

        self.assertEqual(openai.max_tokens, 10, msg="max_tokens must be 10")

        openai.max_tokens = None
        self.assertIsNone(openai.max_tokens, msg="max_tokens must be None")

    def test_bad_temperature(self):
        api_type = "azure"

        with self.assertRaises(TypeError, msg="temperature must be a float"):
            OpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                api_type=api_type,
                temperature="notfloat",
            )

        with self.assertRaises(ValueError, msg="temperature must be between 0 and 2"):
            OpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                api_type=api_type,
                temperature=-1,
            )

    def test_bad_presence_penalty(self):
        api_type = "azure"

        with self.assertRaises(TypeError, msg="presence_penalty must be a float"):
            OpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                api_type=api_type,
                presence_penalty="notfloat",
            )

        with self.assertRaises(
            ValueError, msg="presence_penalty must be between -2 and 2"
        ):
            OpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                api_type=api_type,
                presence_penalty=-3,
            )

    def test_bad_frequency_penalty(self):
        api_type = "azure"

        with self.assertRaises(TypeError, msg="frequency_penalty must be a float"):
            OpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                api_type=api_type,
                frequency_penalty="notfloat",
            )

        with self.assertRaises(
            ValueError, msg="frequency_penalty must be between -2 and 2"
        ):
            OpenAI(
                api_key=self.api_key,
                model_name=self.model_name,
                api_type=api_type,
                frequency_penalty=-3,
            )

    def test_azure_api(self):
        api_type = "azure"
        memory = Memory()
        memory.add_message("user", "Hello!")

        chat_model = OpenAI(
            api_key=self.api_key,
            model_name=self.model_name,
            api_type=api_type,
            api_base=self.api_base,
        )

        response = chat_model(memory, description="You are a chatbot.")
        self.assertIsInstance(response, Response)

    def test_azure_api_with_wrong_api_type(self):
        memory = Memory()
        memory.add_message("user", "Hello!")
        self.api_key = "wrong_api_key"

        chat_model = OpenAI(
            api_key=self.api_key,
            model_name=self.model_name,
        )

        with self.assertRaises(InvalidRequestError):
            chat_model(memory, description="You are a chatbot.")

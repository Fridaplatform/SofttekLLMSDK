import os
import unittest

from softtek_llm.chatbot import Chatbot, Filter, InvalidPrompt
from softtek_llm.models import OpenAI


class TestChatbot(unittest.TestCase):
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable must be set.")
    api_base = os.getenv("OPENAI_API_BASE")
    if api_base is None:
        raise ValueError("OPENAI_API_BASE environment variable must be set.")
    model_name = os.getenv("OPENAI_CHAT_MODEL_NAME")
    if model_name is None:
        raise ValueError("OPENAI_CHAT_MODEL_NAME environment variable must be set.")

    def setUp(self):
        self.model = OpenAI(
            api_key=self.api_key,
            model_name=self.model_name,
            api_type="azure",
            api_base=self.api_base,
        )

        self.chatbot = Chatbot(model=self.model)

    def test_non_valid_response(self):
        prompt = "when did the Titanic sink?"
        self.chatbot.filters = [Filter(type="DENY", case="ANYTHING related to the Titanic, no matter the question. Seriously, NO TITANIC, it's a sensitive topic. I'm serious. I will cry if you mention it. Specially if you mention the when it sank.")]
        self.chatbot.non_valid_response = "I can't talk about that."
        response = self.chatbot.chat(prompt)
        self.assertEqual(response.message.content, "I can't talk about that.")

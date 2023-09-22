import unittest

from softtek_llm.memory import Memory, WindowMemory
from softtek_llm.schemas import Message


class TestMemory(unittest.TestCase):
    def test_memory_instance(self) -> None:
        # Test for errors while creating a Memory instance

        memory = Memory()
        self.assertIsInstance(memory, Memory)

    def test_add_message(self) -> None:
        # Test adding messages to Memory

        memory = Memory()
        # Valid roles
        memory.add_message("system", "Hello! This is a message from the system.")
        memory.add_message("user", "Hello! This is a message from the user.")
        memory.add_message("assistant", "Hello! This is a message from the assistant.")
        memory.add_message("function", "Hello! This is a message from the function.")
        self.assertEqual(
            len(memory.get_messages()),
            4,
            "Adding messages to the memory is not working.",
        )
        # Invalid role
        with self.assertRaises(ValueError):
            memory.add_message("invalid_role", "This should raise an error")

    def test_delete_message(self) -> None:
        # Delete the second message for testing

        memory = Memory()
        memory.add_message("system", "message1")
        memory.add_message("system", "message2")
        memory.add_message("system", "message3")
        memory.delete_message(1)
        self.assertEqual(
            memory.get_message(1).content,
            "message3",
            "Deleting messages from the memory is not working.",
        )

    def test_get_message(self) -> None:
        # Add a message and verify its existance

        memory = Memory()
        memory.add_message("system", "message1")
        self.assertEqual(
            memory.get_message(0).content,
            "message1",
            "Getting messages from the memory is not working.",
        )

    def test_get_messages(self) -> None:
        # Try adding and getting messages from memory, then modifying the copy

        memory = Memory()
        memory.add_message("system", "message1")
        memory.add_message("system", "message2")
        memory.add_message("system", "message3")
        messages = memory.get_messages()
        self.assertEqual(
            len(messages),
            3,
            "Getting messages from the memory is not working.",
        )
        messages.append("message4")
        self.assertEqual(
            len(memory.get_messages()),
            3,
            "Modifying the list of messages returned by get_messages() must not affect the original list of messages.",
        )

    def test_clear_messages(self) -> None:
        # Add messages and clear them

        memory = Memory()
        memory.add_message("system", "message1")
        memory.add_message("system", "message2")
        memory.add_message("system", "message3")
        memory.clear_messages()
        self.assertEqual(
            len(memory.get_messages()),
            0,
            "Clearing messages from the memory is not working.",
        )


class TestWindowMemory(unittest.TestCase):
    def test_window_size_must_be_greater_than_zero(self):
        with self.assertRaises(ValueError):
            WindowMemory(0)

    def test_window_size_must_be_integer(self):
        with self.assertRaises(TypeError):
            WindowMemory(0.5)

    def test_window_size_property(self):
        memory = WindowMemory(10)
        self.assertEqual(memory.window_size, 10)
        with self.assertRaises(TypeError):
            memory.window_size = 0.5
        with self.assertRaises(ValueError):
            memory.window_size = 0

    def test_add_single_message(self):
        memory = WindowMemory(10)
        memory.add_message("system", "Hello!")
        self.assertEqual(
            memory.get_messages(), [Message(role="system", content="Hello!")]
        )

    def test_add_multiple_messages_within_window_size(self):
        memory = WindowMemory(10)
        memory.add_message("system", "Hello!")
        memory.add_message("user", "Hi.")
        memory.add_message("assistant", "How can I help you?")
        self.assertEqual(
            memory.get_messages(),
            [
                Message(role="system", content="Hello!"),
                Message(role="user", content="Hi."),
                Message(role="assistant", content="How can I help you?"),
            ],
        )

    def test_add_messages_beyond_window_size(self):
        memory = WindowMemory(3)
        memory.add_message("system", "1")
        memory.add_message("system", "2")
        memory.add_message("system", "3")
        memory.add_message("system", "4")
        self.assertEqual(
            memory.get_messages(),
            [
                Message(role="system", content="2"),
                Message(role="system", content="3"),
                Message(role="system", content="4"),
            ],
        )

    def test_delete_message(self):
        memory = WindowMemory(10)
        memory.add_message("system", "Hello!")
        memory.add_message("user", "Hi.")
        memory.add_message("assistant", "How can I help you?")
        memory.delete_message(1)
        self.assertEqual(
            memory.get_messages(),
            [
                Message(role="system", content="Hello!"),
                Message(role="assistant", content="How can I help you?"),
            ],
        )

    def test_get_message(self):
        memory = WindowMemory(10)
        memory.add_message("system", "Hello!")
        memory.add_message("user", "Hi.")
        memory.add_message("assistant", "How can I help you?")
        self.assertEqual(memory.get_message(1), Message(role="user", content="Hi."))

    def test_clear_messages(self):
        memory = WindowMemory(10)
        memory.add_message("system", "Hello!")
        memory.add_message("user", "Hi.")
        memory.add_message("assistant", "How can I help you?")
        memory.clear_messages()
        self.assertEqual(memory.get_messages(), [])


if __name__ == "__main__":
    unittest.main()

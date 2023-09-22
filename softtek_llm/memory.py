from typing import List, Literal

from softtek_llm.schemas import Message
from typing_extensions import override


class Memory:
    """
    # Memory
    Represents the memory of the assistant. Stores all the messages that have been exchanged between the user and the assistant.

    ## Methods
    - `add_message`: Adds a message to the memory.
    - `delete_message`: Deletes a message from the memory.
    - `get_message`: Returns a message from the memory.
    - `get_messages`: Returns all the messages from the memory. It is a copy of the original list of messages. Appending to this list will not affect the original list.
    - `clear_messages`: Clears all messages from the memory.
    """

    def __init__(self):
        """Initializes the Memory class."""
        self.__messages: List[Message] = []

    @classmethod
    def from_messages(cls, messages: List[Message]):
        """Initializes the Memory class from a list of messages.

        Args:
            messages (List[Message]): The list of messages to initialize the memory with.

        Returns:
            Memory: The initialized memory.
        """
        memory = cls()
        for message in messages:
            memory.add_message(message.role, message.content)
        return memory

    def add_message(
        self, role: Literal["system", "user", "assistant", "function"], content: str
    ):
        """Adds a message to the memory.

        Args:
            role (Literal["system", "user", "assistant", "function"]): The role of the message.
            content (str): The content of the message.
        """
        self.__messages.append(Message(role=role, content=content))

    def delete_message(self, index: int):
        """Deletes a message from the memory.

        Args:
            index (int): The index of the message to delete.
        """
        self.__messages.pop(index)

    def get_message(self, index: int) -> Message:
        """Returns a message from the memory.

        Args:
            index (int): The index of the message to return.

        Returns:
            Message: The message at the given index.
        """
        return self.__messages[index]

    def get_messages(self) -> List[Message]:
        """Returns all the messages from the memory. It is a copy of the original list of messages. Appending to this list will not affect the original list.

        Returns:
            List[Message]: A copy of the list of messages.
        """
        return self.__messages.copy()

    def clear_messages(self):
        """Clears all messages from the memory."""
        self.__messages.clear()


class WindowMemory(Memory):
    """
    # Window Memory
    Represents the memory of the assistant. Stores all the messages that have been exchanged between the user and the assistant. It has a maximum size.

    ## Attributes
    - `window_size` (int): The maximum number of messages to store in the memory.

    ## Methods
    - `add_message`: Adds a message to the memory.
    - `delete_message`: Deletes a message from the memory.
    - `get_message`: Returns a message from the memory.
    - `get_messages`: Returns all the messages from the memory. It is a copy of the original list of messages. Appending to this list will not affect the original list.
    - `clear_messages`: Clears all messages from the memory.
    """

    @override
    def __init__(self, window_size: int):
        """Initializes the WindowMemory class.

        Args:
            window_size (int): The maximum number of messages to store in the memory.
        """
        super().__init__()
        self.window_size = window_size

    @property
    def window_size(self):
        """The maximum number of messages to store in the memory."""
        return self.__window_size

    @window_size.setter
    def window_size(self, window_size: int):
        if not isinstance(window_size, int):
            raise TypeError("window_size must be an integer.")
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0.")
        self.__window_size = window_size

    @override
    def add_message(
        self, role: Literal["system", "user", "assistant", "function"], content: str
    ):
        """Adds a message to the memory.

        Args:
            role (Literal["system", "user", "assistant", "function"]): The role of the message.
            content (str): The content of the message.
        """
        super().add_message(role, content)
        if len(self.get_messages()) > self.window_size:
            self.delete_message(0)

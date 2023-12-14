# Memory

Represents the memory of the assistant. Stores all the messages that have been exchanged between the user and the assistant.

### Index

- [Memory](#memory)
- [Window Memory](#window-memory)

## Memory

```python
Memory()
```

Represents the memory of the assistant. Stores all the messages that have been exchanged between the user and the assistant.

### Methods

```python
@classmethod
Memory.from_messages(
  messages: List[Message]
)
```

Initializes the `Memory` class from a list of [`messages`](./schemas/message.md).

#### Args

- `messages` (`List[`[`Message`](./schemas/message.md)`]`): The list of messages to initialize the memory with.

#### Returns

- `Memory`: The initialized memory.

```python
add_message(
  role: Literal["system", "user", "assistant", "function"],
  content: str
)
```

Adds a message to the memory.

#### Args

- `role` (`Literal["system", "user", "assistant", "function"]`): The role of the message.
- `content` (`str`): The content of the message.

```python
delete_message(
  index: int
)
```

Deletes a message from the memory.

#### Args

- `index` (`int`): The index of the message to delete.

```python
get_message(
  index: int
) -> Message
```

Returns a [`message`](./schemas/message.md) from the memory.

#### Args

- `index` (`int`): The index of the message to return.

#### Returns

- [`Message`](./schemas/message.md): The message at the given index.

```python
get_messages() -> List[Message]
```

Returns all the messages from the memory. It is a copy of the original list of messages. **Appending to this list will not affect the original list**.

#### Returns

- `List[`[`Message`](./schemas/message.md)`]`: A copy of the list of messages.

```python
clear_messages()
```

Clears all messages from the memory.

```python
messages_to_dict() -> List[Dict]
```

Returns all the messages from the memory in a list of dictionaries.

#### Returns

- `List[Dict]`: The list of messages in dictionary format.

## Window Memory

```python
WindowMemory(
  window_size: int
)
```

Represents the memory of the assistant. Stores all the messages that have been exchanged between the user and the assistant. It has a **maximum size**. It extends the [`Memory`](#memory-1) class.

#### Args

`window_size` (`int`): The maximum number of messages to store in the memory.

### Properties

- `window_size`: The maximum number of messages to store in the memory.

### Methods

```python
@classmethod
WindowMemory.from_messages(
  messages: List[Message],
  window_size: int
)
```

Initializes the `WindowMemory` class from a list of [`messages`](./schemas/message.md).

#### Args

- `messages` (`List[`[`Message`](./schemas/message.md)`]`): The list of messages to initialize the memory with.
- `window_size` (`int`): The maximum number of messages to store in the memory.

#### Returns

- `WindowMemory`: The initialized memory.

```python
add_message(
    role: Literal["system", "user", "assistant", "function"],
    content: str
)
```

Adds a message to the memory. If the memory is full, **the oldest message will be deleted**.

#### Args
- `role` (`Literal["system", "user", "assistant", "function"]`): The role of the message.
- `content` (`str`): The content of the message.

```python
delete_message(
  index: int
)
```

Deletes a message from the memory.

#### Args

- `index` (`int`): The index of the message to delete.

```python
get_message(
  index: int
) -> Message
```

Returns a [`message`](./schemas/message.md) from the memory.

#### Args

- `index` (`int`): The index of the message to return.

#### Returns

- [`Message`](): The message at the given index.

```python
get_messages() -> List[Message]
```

Returns all the messages from the memory. It is a copy of the original list of messages. **Appending to this list will not affect the original list**.

#### Returns

- `List[`[`Message`](./schemas/message.md)`]`: A copy of the list of messages.

```python
clear_messages()
```

Clears all messages from the memory.

```python
messages_to_dict() -> List[Dict]
```

Returns all the messages from the memory in a list of dictionaries.

#### Returns

- `List[Dict]`: The list of messages in dictionary format.

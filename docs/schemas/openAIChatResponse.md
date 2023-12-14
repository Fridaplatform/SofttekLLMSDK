# OpenAI Chat Response

A model class for an OpenAI chat response.

## Attributes

- `id` (`str`): A unique identifier for the chat completion.
- `object` (`Literal["chat.completion"]`): The object type, which is **always `chat.completion`**.
- `created` (`int`): The Unix timestamp (in seconds) of when the chat completion was created.
- `model` (`str`): The model used for the chat completion.
- `choices` (`List[`[`OpenAIChatChoice`](openAIChatChoice.md)`]`): A list of chat completion choices. Can be more than one if `n` is greater than `1`.
- `usage` ([`Usage`](usage.md)): Usage statistics for the completion request.

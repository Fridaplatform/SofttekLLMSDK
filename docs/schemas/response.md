# Response

Represents the response from an inference API.

## Attributes

- `message` ([`Message`](message.md)): Message object that the API generated as a response.
- `created` (`int`): Unix timestamp for when the response was created.
- `latency` (`int`): Time in milliseconds taken to generate the response.
- `from_cache` (`bool`): Whether the response was served from [cache](../cache.md) or not.

## Optional Attributes

- `model` (`str`): String representation of the model used to generate the response. Defaults to `""`.
- `usage` ([`Usage`](usage.md)): Usage object containing metrics of resource usage from generating the response. Defaults to [`Usage`](usage.md)`()`.
- `additional_kwargs` (`Dict`): Optional dictionary of additional keyword arguments. Defaults to `{}`.

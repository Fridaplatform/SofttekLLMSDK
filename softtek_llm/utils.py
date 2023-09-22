import openai


def setup_azure(api_base: str | None, api_version: str):
    """Sets up the Azure API.

    Args:
        api_base (str | None): Base URL for Azure API.
        api_version (str): API version for Azure API.

    Raises:
        ValueError: When api_base is None.
    """
    if api_base is None:
        raise ValueError("api_base must be specified for Azure API.")
    openai.api_base = api_base
    openai.api_version = api_version

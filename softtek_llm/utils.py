from openai import AzureOpenAI
import unicodedata


def setup_azure(api_key: str | None, api_base: str | None, api_version: str):
    """Sets up the Azure API.

    Args:
        `api_key` (str | None): API Key for Azure API.
        `api_base` (str | None): Base URL for Azure API.
        `api_version` (str): API version for Azure API.

    Raises:
        ValueError: When api_key is None.
        ValueError: When api_base is None.

    """

    if api_key is None:
        raise ValueError("api_key must be specified for Azure API.")
    
    if api_base is None:
        raise ValueError("api_base must be specified for Azure API.")
    
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=api_base,
    )


def strip_accents_and_special_characters(text: str) -> str:
    """
    Strips accents and special characters from the given text.

    Args:
        `text` (str): The input text to strip accents and special characters from.

    Returns:
        str: The input text with accents and special characters removed.
    """
    
    text = unicodedata.normalize("NFD", text)
    text = text.encode("ascii", "ignore")
    text = text.decode("utf-8")
    return str(text)
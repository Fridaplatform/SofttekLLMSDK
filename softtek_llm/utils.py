import openai
import unicodedata


def setup_azure(api_base: str | None, api_version: str):
    """Sets up the Azure API.

    Args:
        `api_base` (str | None): Base URL for Azure API.
        `api_version` (str): API version for Azure API.

    Raises:
        ValueError: When api_base is None.
    """
    if api_base is None:
        raise ValueError("api_base must be specified for Azure API.")
    openai.api_base = api_base
    openai.api_version = api_version


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
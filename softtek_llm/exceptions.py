class InvalidPrompt(Exception):
    """Raised when the prompt is invalid."""

    pass


class TokensExceeded(Exception):
    """Raised when the tokens are exceeded."""

    pass

class KnowledgeBaseEmpty(Exception):
    """Raised when the knowledge base is empty."""

    pass
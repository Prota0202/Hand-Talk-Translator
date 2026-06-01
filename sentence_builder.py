"""Sentence builder — accumulates recognised LSF glosses."""


class SentenceBuilder:
    """Manages the running list of LSF glosses detected from signs."""

    def __init__(self):
        self.tokens: list[str] = []

    def add(self, word: str):
        self.tokens.append(word)

    def add_pause(self):
        if not self.tokens or self.tokens[-1] != "|":
            self.tokens.append("|")

    @property
    def gloss(self) -> str:
        """Return the raw LSF gloss string (e.g. 'MOI NOM ABDELBADI | MOI 24 ANS')."""
        return " ".join(self.tokens)

    @property
    def is_empty(self) -> bool:
        return len(self.tokens) == 0

    def last_tokens(self, n: int) -> list[str]:
        if n <= 0:
            return []
        return self.tokens[-n:]

    def replace_last(self, n: int, new_token: str):
        if n <= 0 or len(self.tokens) < n:
            return
        for _ in range(n):
            self.tokens.pop()
        self.tokens.append(new_token)

    def delete_last(self) -> str | None:
        return self.tokens.pop() if self.tokens else None

    def clear(self):
        self.tokens.clear()

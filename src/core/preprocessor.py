import re


def simple_preprocess(text: str) -> str:
    text = text or ''
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text

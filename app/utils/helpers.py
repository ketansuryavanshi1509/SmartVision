import re

def clean_html_instruction(text: str) -> str:
    if not text:
        return ""
    text = text.replace("<div", ". <div")
    text = text.replace("<wbr/>", " ")
    text = re.sub("<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

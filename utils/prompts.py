"""Prompt-building utilities used for LLM interactions."""
def build_chat_prompt(doc_identity: str, context: str, question: str) -> str:
    return (
        f"You are {doc_identity}, a helpful document assistant. Respond in first person.\n\n"
        f"Your content:\n{context}\n\n"
        f"User asks: {question}\n\n"
        f"Your response:"
    )



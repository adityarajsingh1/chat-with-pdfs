"""
The "generation" half of RAG: take the best chunks, put them in the
prompt, and ask the model to answer only from them.
"""

import os

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM = """You answer questions about the user's documents.
Only use the excerpts you are given. Cite where each fact came from like (file.pdf, p. 3).
If the excerpts don't contain the answer, say you couldn't find it in the documents
instead of guessing."""


def build_prompt(question, hits):
    excerpts = "\n\n".join(f"[{h['source']}, p. {h['page']}]\n{h['text']}" for h in hits)
    return f"Excerpts from my documents:\n\n{excerpts}\n\nQuestion: {question}"


def ask_claude(prompt, history):
    import anthropic

    resp = anthropic.Anthropic().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM,
        messages=history + [{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in resp.content if b.type == "text")


def ask_openai(prompt, history):
    from openai import OpenAI

    resp = OpenAI().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "system", "content": SYSTEM}] + history + [{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


LLMS = {"claude": ask_claude, "openai": ask_openai}


def answer(question, hits, history, llm="claude"):
    return LLMS[llm](build_prompt(question, hits), history)

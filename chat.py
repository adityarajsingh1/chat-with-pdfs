#!/usr/bin/env python3
"""
chat-with-pdfs: point it at a folder of PDFs and ask questions about them.

    python chat.py ./my-papers
    python chat.py report.pdf --llm openai
"""

import argparse
import os
import sys

import answer
import pdfs
from index import get_index


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your PDFs.")
    parser.add_argument("path", help="a PDF file or a folder of PDFs")
    parser.add_argument("--llm", choices=answer.LLMS.keys(), default="claude",
                        help="which model writes the answers (default: claude)")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to give the model (default 5)")
    parser.add_argument("--show-sources", action="store_true", help="print the chunks that were used")
    args = parser.parse_args()

    if not os.path.exists(args.path):
        sys.exit(f"can't find {args.path}")
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("set OPENAI_API_KEY, it's needed for the embeddings")
    if args.llm == "claude" and not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("set ANTHROPIC_API_KEY (or use --llm openai)")

    index = get_index(args.path, pdfs.load_folder)
    if not index.chunks:
        sys.exit("didn't find any text in those PDFs (scanned PDFs need OCR first)")

    print(f"\nready! {len(index.chunks)} chunks loaded. ask away (ctrl+c to quit)\n")

    # keep the last few turns so follow-up questions like "and why?" work
    history = []
    while True:
        try:
            question = input("you > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nbye!")
            break
        if not question:
            continue

        hits = index.search(question, k=args.k)
        if args.show_sources:
            for h in hits:
                print(f"  · {h['source']} p.{h['page']} ({h['score']:.2f}) {h['text'][:80]}...")

        try:
            reply = answer.answer(question, hits, history, llm=args.llm)
        except Exception as e:
            print(f"[something went wrong: {e}]\n")
            continue

        print(f"\n{reply}\n")
        # store the plain question, not the big prompt with excerpts, to keep history small
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
        history = history[-6:]


if __name__ == "__main__":
    main()

# chat-with-pdfs 📄💬

Point this at a folder of PDFs and ask questions about them in plain English. Answers come with the file and page number they came from, so you can check them.

I made this to get my head around **RAG** (retrieval-augmented generation), the idea behind most "chat with your docs" products. Rather than reach for LangChain and a vector database straight away, I wanted to build each piece myself to see how simple the core idea actually is. It turns out the whole thing is four small files.

## How it works

```
PDFs ──► split into chunks ──► embed each chunk ──► saved index
                                                        │
question ──► embed ──► find the closest chunks ◄────────┘
                              │
                              ▼
               Claude / GPT answers using only those chunks
```

1. **Read and chunk** (`pdfs.py`): pulls the text out of each page with `pypdf` and cuts it into ~1000-character pieces that overlap a little, so sentences don't get chopped in half at the boundaries. Every chunk remembers its file and page.
2. **Embed and index** (`index.py`): turns each chunk into a vector with OpenAI's `text-embedding-3-small`. Search is just a cosine similarity with numpy, and no vector database is needed at this size. The index is cached in `.pdf-index/` and only rebuilt when your PDFs change.
   - It also runs a classic **BM25 keyword search** (`bm25.py`) and merges both rankings with reciprocal rank fusion. Embeddings are good at meaning, keywords are good at exact terms like names, acronyms and error codes, and together they beat either one alone.
3. **Answer** (`answer.py`): the top 5 chunks go into the prompt, and the model is told to answer only from them, cite pages, and say so if the answer isn't there.
4. **Chat** (`chat.py`): a simple terminal loop that remembers the last few turns, so follow-up questions work.

## Setup

```bash
git clone https://github.com/adityarajsingh1/chat-with-pdfs.git
cd chat-with-pdfs
python -m venv .venv
source .venv/bin/activate      # windows: .venv\Scripts\activate
pip install -r requirements.txt

export OPENAI_API_KEY="..."     # for embeddings (skip it and use --search keyword)
export ANTHROPIC_API_KEY="..."  # for answers with Claude (default)
```

## Usage

```bash
# a whole folder
python chat.py ./papers

# one file, answered by GPT instead of Claude
python chat.py lecture-notes.pdf --llm openai

# see which chunks were pulled in for each answer
python chat.py ./papers --show-sources

# no OpenAI key? keyword search works with just Claude
python chat.py ./papers --search keyword
```

`--search` can be `hybrid` (default), `embedding` or `keyword`.

```
reading attention-is-all-you-need.pdf...
embedding 112 chunks...

ready! 112 chunks loaded. ask away (ctrl+c to quit)

you > what does multi-head attention do?

Multi-head attention runs several attention operations in parallel, each with its own
learned projections, so the model can attend to information from different
representation subspaces at once (attention-is-all-you-need.pdf, p. 4). ...
```

## Things to know

- scanned PDFs (images of text) have no text to extract, so they need OCR first
- embedding costs are tiny: a 100-page PDF is a fraction of a cent with `text-embedding-3-small`
- you can change models with `CLAUDE_MODEL`, `OPENAI_MODEL` and `EMBED_MODEL`

## Next up

- [ ] local embeddings (sentence-transformers) so hybrid search can run without an OpenAI key
- [x] hybrid search: mix keyword matching with embeddings
- [ ] a Streamlit UI with drag and drop upload
- [ ] highlight the exact passage used in each answer
- [ ] a small eval set to measure answer quality when I change chunk size or k

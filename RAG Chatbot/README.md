# RAG Chatbot

Ask questions about your own documents and get answers that cite the passages
they came from — or an honest "not in your documents" when they don't.

**Stack:** Gemini for chat · `sentence-transformers/all-MiniLM-L6-v2` for
embeddings (local, no API) · Postgres + pgvector for storage.

```
┌──────────── ingest ────────────┐        ┌──────────── ask ─────────────┐
 upload → hash (dedupe) → 202                question + history
        ↓ background                              ↓ condense
   load → chunk (+ page no.)               standalone question
        ↓                                         ↓ embed (MiniLM, local)
   embed (MiniLM, local)                    pgvector cosine search
        ↓                                         ↓
   Postgres + pgvector  ←──────────────────  top-k above threshold
                                                  ↓ no → refuse (no LLM call)
                                            Gemini → streamed answer + citations
```

---

## Quick start

**1. Start Postgres** (pgvector is already compiled into the image):

```bash
docker compose up -d
```

Or point `DATABASE_URL` at any Postgres where you can run
`CREATE EXTENSION vector`.

**2. Install.** PyTorch comes as a transitive dependency of
sentence-transformers; install the CPU-only build first or pip will pull several
gigabytes of CUDA wheels you don't need:

```bash
cd "RAG Chatbot"
python -m venv .venv
.venv\Scripts\activate                # Windows
# source .venv/bin/activate           # macOS / Linux

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt
```

**3. Configure.** Copy `.env.example` to `.env`; only two values are required:

```ini
DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
GEMINI_API_KEY=your-key-here          # free: https://aistudio.google.com/apikey
```

**4. Run:**

```bash
cd backend
uvicorn app.main:app --reload
```

Open **http://localhost:8000**. The frontend is served by the same app, so there
is no second server and no CORS to configure. First boot downloads ~90 MB of
model weights and creates the database schema itself — no migration step.

---

## Notes on the three choices

### Gemini for chat

No weights on disk, no GPU, no RAM — the memory budget goes to the embedding
model instead. The free tier needs no credit card.

⚠️ **Gemini's free tier uses your content to improve Google's products.** Fine
for a demo; don't send confidential documents through it. Free-tier daily caps
surface as a clean `RateLimitError` pointing at
[your rate-limit dashboard](https://aistudio.google.com/rate-limit).

### all-MiniLM-L6-v2 for embeddings

No API key, no quota, no per-request cost — ingesting a large corpus is bounded
by CPU rather than a daily request cap. 384 dimensions keeps the index compact
(~1.5 KB per vector including overhead).

It is also **symmetric**: one encoder for both passages and questions. That
removes the document/query task-type distinction that API embeddings require and
that quietly costs recall when you get it wrong.

The cost is memory. `sentence-transformers` pulls PyTorch:

| | |
|---|---|
| Installed dependencies | ~1 GB (CPU-only wheel; ~3 GB+ with CUDA) |
| Resident memory, model loaded | ~400–600 MB |
| Model weights | ~90 MB |

**This does not fit a 512 MB instance.** Render's free tier will OOM. See
[Deploying](#deploying) for the options.

### Postgres + pgvector

One database for documents, chunks, vectors and conversations. `asyncpg`
directly rather than an ORM — the schema is four tables of hand-written SQL, and
going driver-level sidesteps the friction between SQLAlchemy's asyncpg dialect,
pgvector's type codec, and PgBouncer.

**The vector dimension is read from the loaded model, not from config.** There is
no `EMBED_DIM` setting to drift out of step with the column.

---

## What makes this more than naive RAG

Four things that basic tutorials skip, each of which fails *quietly*:

1. **Follow-up questions are condensed.** "What about the second one?" embeds
   into a meaningless vector. Before retrieval, the conversation and the new
   message are rewritten into one standalone question.
2. **A similarity threshold guards the LLM.** If nothing clears
   `SIMILARITY_THRESHOLD`, the app refuses *without calling the model*. Handed
   irrelevant passages, an LLM will confidently invent an answer from them.
3. **Deleting a document deletes its vectors** (`ON DELETE CASCADE`). Otherwise
   removed documents keep appearing in citations forever.
4. **Chunks carry their page number**, so a citation points somewhere real.

Plus: SHA-256 content hashing so re-uploads cost nothing, background ingestion
so uploads return instantly, and a `failed` status carrying the real error
instead of a document stuck on `processing`.

---

## Deploying

The app is a single container: FastAPI serving both the API and the frontend,
plus an external Postgres.

**Database.** Use a managed instance — [Neon](https://neon.com) has a free tier
with pgvector and no inactivity-deletion policy. Use the **pooled** endpoint
(`-pooler` in the hostname). Two to avoid: **Render's own free Postgres expires
30 days after creation**, and **Supabase pauses free projects after 7 days idle**.

Paste the connection string exactly as given — the app strips the `sslmode` and
`channel_binding` parameters asyncpg rejects.

**App.** It needs **≥1 GB of RAM** because of PyTorch. Options:

| Option | Notes |
|---|---|
| **Render Starter** (~$7/mo) | [render.yaml](render.yaml) is configured for this. Also removes the free tier's 15-minute spin-down. |
| **Google Cloud Run** | Deploy [Dockerfile](Dockerfile). Always-free tier allows 1 GB instances; requires a billing account. |
| **Any VPS / Fly / Railway** | Same Dockerfile, ≥1 GB instance. |
| **Stay on a free 512 MB tier** | Only by switching embeddings to an API (Gemini, Cohere, Jina) so nothing but an HTTP client lives in memory. That is a code change, not a config one. |

The [Dockerfile](Dockerfile) bakes the model weights into the image — otherwise
every cold start on an ephemeral filesystem re-downloads them.

---

## Configuration

Every setting is documented in [.env.example](.env.example). The ones that matter:

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | — | **Required.** |
| `GEMINI_API_KEY` | — | **Required.** |
| `SIMILARITY_THRESHOLD` | `0.35` | Cosine floor. For MiniLM, unrelated text scores ~0.0–0.2 and a real match ~0.4–0.7. Raise it if weak matches produce confident answers; lower it if it refuses too eagerly. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Overlap stops facts being cut in half at a boundary. |
| `TOP_K` | `8` | Passages retrieved per question. |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Any sentence-transformers model. **Changing it requires a re-ingest** — see below. |
| `EMBEDDING_DEVICE` | `cpu` | Set `cuda` if you have a GPU. |

### Changing the embedding model

`chunks.embedding` is `VECTOR(n)` and **a pgvector column's dimension is fixed at
creation.** To switch to a model with a different output size:

```sql
DROP TABLE chunks;
```

then restart and re-upload. Startup compares the live column against the loaded
model and refuses to run on a mismatch rather than corrupting the index quietly.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload (multipart). Returns **202**; ingestion runs in the background. |
| `GET` | `/api/documents` | List with `status` and `chunk_count`. Poll this. |
| `DELETE` | `/api/documents/{id}` | Delete the document and its vectors. |
| `POST` | `/api/chat` | Ask a question. **SSE stream.** |
| `POST` | `/api/chat/sync` | Same, buffered — handy for curl. |
| `GET` | `/healthz` | Liveness plus the active model/database config. |
| `GET` | `/docs` | Interactive OpenAPI docs. |

SSE events: `meta` (conversation id, rewritten question) → `citations` (sent
*before* the text, so sources render as the answer streams) → `token` … →
`done`. `error` replaces the tail if generation fails.

```bash
curl -s localhost:8000/api/chat/sync \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the refund window?"}' | python -m json.tool
```

---

## Layout

```
backend/app/
  main.py              app factory, lifespan, /healthz, static mount
  core/config.py       every tunable, one place
  api/                 documents.py · chat.py · deps.py
  db/                  base.py (protocol) · database.py (Postgres) · models.py
  rag/
    loader.py          PDF / DOCX / text → (page, text)
    chunker.py         recursive split, page-preserving   ← pure, unit-tested
    embeddings.py      sentence-transformers, loaded once
    llm.py             Gemini chat + streaming
    retriever.py       search → threshold → citations
    prompt.py          system prompts, context formatting ← pure, unit-tested
    pipeline.py        orchestration; the only module the API talks to
  tests/               53 offline tests + 15 Postgres integration tests
frontend/              vanilla HTML/CSS/JS, no build step
scripts/init_db.sql    idempotent schema, applied at startup
docker-compose.yml     local Postgres + pgvector
```

Startup order is deliberate: the embedding model loads first because it declares
the vector dimension, then the store is built from that.

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
pytest                        # 53 pass, 15 skipped without a database
ruff check app tests
ruff format --check app tests
```

The default run needs **no database, no network and no API key** — the store,
the embedding model and Gemini are substituted, while the loader, chunker,
retriever, prompt builder and pipeline are production code.

To also exercise the real Postgres store:

```bash
export TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag_test
pytest
```

⚠️ Those tests **drop and recreate the tables** — point them at a scratch
database. CI runs them against a `pgvector/pgvector:pg17` service container,
which is what keeps the in-memory test double honest about real database
behaviour.

## Licence

MIT — see [LICENSE](LICENSE).

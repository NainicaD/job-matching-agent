# job-matching-agent

Match a consolidated resume profile against job descriptions — either with the
Claude API, or entirely offline for free.

The two matching engines are **independent implementations**, not one wrapping
the other. API mode is a real integration with the Anthropic Python SDK. Local
mode is a self-contained TF-IDF matcher that imports nothing from the API path
and runs with no key, no network, and no third-party packages.

---

## Privacy

Nothing personal is in this repository. `resumes/` and the generated
`profile.txt` are both gitignored — clone it, drop your own resumes into
`resumes/`, and run `build-profile` to generate your own profile. Contact
details are redacted from that file by default.

## Setup

You need Python **3.10 or newer** (the `anthropic` 1.x SDK requires it).

```bash
cd job-matching-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### The API key

API mode needs `ANTHROPIC_API_KEY`. Get one from
[console.anthropic.com](https://console.anthropic.com/settings/keys), then:

```bash
# this shell only
export ANTHROPIC_API_KEY='sk-ant-...'

# or permanently (zsh)
echo "export ANTHROPIC_API_KEY='sk-ant-...'" >> ~/.zshrc && source ~/.zshrc
```

Verify it before doing anything else — this makes one deliberately tiny
(`max_tokens=1`) call and reports what it cost:

```bash
python match.py check
```

If the key is missing, `check` prints these instructions instead of failing at
the first real API call. If you would rather not use an environment variable,
`ant auth login` stores an OAuth profile that the SDK picks up automatically.

**Local mode needs none of this.** `python match.py match job.txt --local`
works with `ANTHROPIC_API_KEY` unset and the `anthropic` package uninstalled.

---

## Usage

```bash
python match.py check                              # verify credentials
python match.py build-profile                      # ./resumes/ -> profile.txt
python match.py index-postings                     # ./job_postings/ -> vector store
python match.py rank                               # rank all postings, no API calls
python match.py match --top-k 5                    # retrieval + API reasoning on top 5
python match.py match --top-k 5 --dry-run          # retrieval + payloads, send nothing
python match.py match --top-k 5 --local            # retrieval + offline reasoning
python match.py match job.txt                      # API mode on one explicit file
python match.py match job.txt --dry-run            # inspect the payload, send nothing
python match.py match job.txt --local              # offline, free
python match.py match job.txt --model claude-sonnet-5
python match.py models                             # model IDs and prices
```

### 1. Build the profile

```bash
python match.py build-profile
```

Scans `./resumes/` for `.pdf`, `.tex` and `.txt` files, extracts the text
(`pdfplumber` for PDFs, a LaTeX stripper for `.tex`), and consolidates
everything into `profile.txt`, plus `profile.review.md` for the things that
need a human eye.

What it does beyond concatenating:

- **Rejoins wrapped lines.** PDF extraction emits one line per *visual* line, so
  a three-line bullet arrives as three fragments. Those are rejoined into
  logical bullets before anything else runs.
- **Repairs extraction damage.** Some PDF fonts map the `fi`/`fl` ligature to
  the wrong code point ("financial" → "kinancial"), and some PDFs drop space
  characters entirely ("coordinatingcrossfunctional"). Both are repaired, gated
  on a vocabulary built from your own corpus plus the system word list, so real
  words are never rewritten.
- **Removes exact duplicates** — 40 resume versions share most of their bullets.
- **Flags near-duplicates instead of merging them.** Two phrasings of the same
  project are two pieces of real data, so both stay in `profile.txt` and the
  cluster is listed in `profile.review.md` for you to decide about.
- **Redacts contact details** by default. Your email and phone add nothing to
  matching, and `profile.txt` is what gets sent to the API. Use
  `--keep-contact` to keep them.

The build is **idempotent**: re-running it on an unchanged `./resumes/` produces
a byte-identical `profile.txt`. There is deliberately no timestamp in the
header — a content hash is used instead — since a timestamp would make the file
differ on every run by construction.

### 2. Match against a job

Put the job description in a text file (or pipe it in with `-`):

```bash
python match.py match jobs/example_ml_engineer.txt
```

---

## The retrieval stage (RAG)

Matching one job at a time is fine. Matching fifty is not: at ~$0.03 a call
that is $1.50 to discover that forty of them were never a fit. The retrieval
stage puts a free, local filter in front of the paid reasoning stage.

```
job_postings/*.txt
    |  load + chunk          (LangChain)
    v
embed each chunk             (sentence-transformers, local, free)
    |
    v
Chroma vector store on disk  <-- persisted; unchanged postings are never re-embedded
    |  similarity search, once per profile chunk
    v
ranked postings -> top-k ----> the existing reasoning stage (the only paid step)
```

Nothing above the last arrow costs anything or needs an API key.

### Commands

```bash
python match.py index-postings     # embed ./job_postings/ into ./chroma_db/
python match.py rank               # print the ranking, make no API calls
python match.py match --top-k 5    # rank, then reason over the best 5
```

`rank` prints every posting with its score, so the filtering decision is
visible before it is acted on:

```
   #  score  posting                                               chunks
  --------------------------------------------------------------------------
-> 1   78.9  SAP GTS Consultant - Arbor Consulting Partners        5/5
-> 2   70.7  Teaching Assistant, Data Structures and Machine L...  4/4
-> 3   64.2  Graduate Research Assistant, Applied Machine Lear...  4/4
-> 4   63.0  NLP Engineer, Retrieval & Generation - Corvus AI      5/5
-> 5   61.5  Data Scientist, Clinical Analytics - Meridian Hea...  5/5
   6   57.4  Machine Learning Engineer, Intern (Summer 2026) -...  5/5
  ...
  12   34.9  Registered Nurse, Intensive Care Unit - St. John ...  3/3

  top 5 selected for reasoning; 7 posting(s) filtered out before any API call
```

`match --top-k 5` prints that same table, then runs the **existing** API-mode
matcher on the five selected postings — same request construction, same error
handling, same JSON validation, same cost tracking and session total. The
reasoning stage was not rewritten; it just receives fewer inputs.

The same is true of `--dry-run` and `--local`: both run retrieval first, then
feed the top-k into the path they already had.

### What the pre-ranking saves

12 postings, `claude-haiku-4-5`, profile at ~23k input tokens:

| | API calls | Cost |
|---|---|---|
| Every posting | 12 | ~$0.086 |
| `--top-k 5` | 5 | ~$0.051 |
| `--top-k 3` | 3 | ~$0.041 |
| `rank` only | 0 | $0.00 |

The saving grows with the corpus, and with prompt caching the marginal call is
cheap — so the real win is at 50+ postings, where retrieval turns "$1.50 and
five minutes" into "$0.05 and five seconds". Retrieval itself costs about two
seconds of CPU for 12 postings and is then cached on disk.

### How it works, and why each step is there

**Chunking** (`jobmatch/postings.py`). An embedding model compresses its whole
input into one fixed-length vector, and `all-MiniLM-L6-v2` truncates at 256
word-pieces. Embedding a 2,000-word posting whole does not fail loudly — it
silently encodes the opening and discards the requirements section. Chunking
keeps the tail of the document in the index at all. It also avoids dilution: one
vector for a long document is an average of everything in it, so a posting that
is 90% boilerplate produces a boilerplate-shaped vector.

`chunk_size=400` was chosen by measurement, not taste. At 800 these postings
produced 2–3 chunks each, which quietly broke the scoring below (it averages a
posting's best 3 chunks — with 2 chunks that is just "all of them", and the
dilution problem returns). At 400 each posting yields 3–5 chunks. On this corpus
that moved a strongly-matching NLP posting from 6th to 4th and pushed an
unrelated infrastructure posting from 8th to 10th.

`chunk_overlap=80` repeats each chunk's tail at the head of the next, so a split
landing mid-phrase — `"experience with transformer"` | `"architectures"` — does
not lose text that only means something whole.

**The profile is chunked too**, for the same truncation reason, and it matters
more: `profile.txt` is ~90 KB, so embedding it whole would encode roughly its
first 200 words and discard every project below that.

**Embedding + store** (`jobmatch/retrieval.py`). `HuggingFaceEmbeddings` wraps
sentence-transformers behind LangChain's `Embeddings` interface
(`embed_documents` for indexing, `embed_query` for searching). The Chroma
collection is created with `hnsw:space=cosine` — Chroma defaults to squared L2,
which is wrong for sentence embeddings — so the distance it returns is
`1 - cosine_similarity` and converts back with one subtraction. That conversion
is done explicitly rather than via `similarity_search_with_relevance_scores`, so
the arithmetic is visible.

**Search and aggregation.** One similarity search per *profile* chunk, not one
for the whole profile. Search returns chunks, but the decision is about
postings, so chunk scores are combined — and the choice matters:

- `max` is too generous: nearly every posting has one "strong communication
  skills" paragraph that matches any profile, so unrelated postings score high.
- mean over *all* chunks punishes long postings: a great match padded with
  benefits boilerplate scores below a short mediocre one.
- **mean of the best 3** requires that several parts of the posting connect to
  the profile without punishing filler. Same intuition as `recall@k` — one hit
  is noise, a cluster is signal.

`tests/test_retrieval.py` asserts all three of those properties directly.

**Incremental indexing.** Embedding is the slow part, so each stored chunk
carries an *index key* covering the posting's content hash, the chunk size and
overlap, and the model name. Re-running `index-postings` re-embeds only what
changed. All three inputs are in the key deliberately: hashing just the text was
a bug, because changing `--chunk-size` then left every key identical and the
store went on answering queries from vectors built under the old settings.

### Scores are a sort key, not a percentage

Retrieval scores are only comparable *to each other within one ranking*. They
say "spend your calls here first", not "you are a 78% fit". The reasoning stage
produces the actual judgement.

### The zero-network fallback

The embedding backend needs a one-time ~90 MB model download. If you are offline,
or would rather not install torch:

```bash
python match.py rank --retrieval-backend tfidf
python match.py match --top-k 5 --local --retrieval-backend tfidf
```

That path reuses the stdlib TF-IDF already in `local_matcher` — no LangChain, no
Chroma, no model, no vector store (recomputing TF-IDF on a corpus this size
costs milliseconds, so persistence would be complexity for nothing). It is a
genuinely weaker ranker, and the difference is instructive: on this corpus the
lexical backend ranks the NLP Engineer posting 8th because the profile says
"information retrieval" and "transformer" where the posting says "semantic
search" and "embedding pipelines". The embedding backend ranks it 4th. That gap
*is* what embeddings buy you.

### A note on what retrieval sees

`profile.txt` consolidates every resume version in `./resumes/`. If those target
very different roles — ML engineering, research, program coordination — the
merged profile is genuinely a generalist one, and the ranking reflects that
rather than your current focus. If you want retrieval to rank for one direction,
build a profile from a narrower set of resumes:

```bash
python match.py build-profile --resumes ./resumes_ml --out profile.ml.txt
python match.py rank --profile profile.ml.txt
```

---

## The three modes

| | API mode (default) | `--dry-run` | `--local` |
|---|---|---|---|
| What it is | Claude reads both documents and judges | Prints the request, sends nothing | TF-IDF + cosine similarity |
| Needs a key | yes | no | **no** |
| Needs network | yes | no | **no** |
| Cost | ~$0.03/job on Haiku | $0.00 | $0.00 |
| Understands synonyms | yes | — | no |
| Good for | the shortlist | tuning the prompt | ranking 50 jobs cheaply |

### API mode

Sends the profile and the job description to `client.messages.create()` and asks
for a structured assessment as strict JSON: a score, matching strengths with
evidence quoted from your profile, gaps with severities, and a short rationale.

Defaults to **`claude-haiku-4-5`** — the cheapest current model, which is the
right default while iterating. Override with `--model` (`python match.py models`
lists the known IDs and prices).

Every call prints its token usage and cost, and a multi-job run prints a session
total:

```
  tokens: in 23,234 | out 412 | cache-write 23,100   cost: $0.0542
  session: 2 call(s) | in 23,384 | out 812
           cache: 23,100 written, 23,100 read (reads bill at 10% of input)
           total cost: $0.0586
```

Spend is also appended to `.jobmatch_usage.jsonl` so you can total it later
(`--no-ledger` to disable).

**Prompt caching.** The profile is the large, unchanging part of the request, so
it goes in `system` with `cache_control: ephemeral` while only the job
description varies. The first job in a session pays to write the cache; every
later job re-reads it at 10% of the input rate. Matching five jobs costs far
less than five times one job. `--no-cache` turns this off.

**Error handling.** Every failure mode is handled and explained rather than
raised as a traceback: a missing or invalid key, a model you lack access to, a
typo'd model ID, a 429 (retried, honouring the server's `retry-after`), 5xx and
network failures (retried with exponential backoff and jitter), a response that
is not valid JSON, a response truncated by `max_tokens`, and a refusal.

### Dry-run mode

```bash
python match.py match jobs/example_ml_engineer.txt --dry-run
```

Prints the exact system prompt, messages, model and parameters that *would* be
sent, plus an estimated input token count and cost — without calling the API.
Long blocks are elided in the middle; `--full` prints them verbatim.

This is built from the same `build_request()` function the real call uses, and a
test asserts the two payloads are identical, so what you inspect is what would
be sent. `--count-tokens` swaps the ~4-chars-per-token estimate for an exact
count from the free `count_tokens` endpoint (that one *does* make a network
call).

### Local mode

```bash
python match.py match jobs/example_ml_engineer.txt --local
```

Fully offline. Splits `profile.txt` into one document per bullet, computes
TF-IDF over unigrams and bigrams, and scores the job description against it with
cosine similarity. Output uses the same layout as API mode, under a banner that
says explicitly it is not an LLM judgement.

Three backends, picked at runtime: `sentence-transformers` if you pass
`--embeddings`, `scikit-learn` if installed, otherwise a pure-Python TF-IDF
implementation that needs only the standard library. `--no-sklearn` forces the
pure-Python path.

**Read the score correctly.** The headline number is the share of the job's most
distinctive terms that your profile evidences, **weighted by how distinctive each
term is** — missing the job's single most characteristic term costs more than
missing its thirtieth. It is not a cosine similarity, which sits in the 0.1–0.4
range even for an excellent match and reads as alarming next to an LLM's 0–100.
The raw cosine is reported alongside it.

Terms are stemmed before matching, so a job asking you to "coordinate" programs
matches a profile full of "coordination" and "coordinating" instead of reporting
a false gap.

Even so, local mode matches **strings, not meaning**. If the job says
"recommender systems" and your resume says "collaborative filtering", local mode
calls that a gap and API mode does not. Use local mode to rank a long list
cheaply, then spend API tokens on the top of that ranking.

---

## Cost

With the profile at ~23k input tokens:

| Model | First job | Each later job in the same session |
|---|---|---|
| `claude-haiku-4-5` (default) | ~$0.032 | ~$0.005 |
| `claude-sonnet-5` | ~$0.064 | ~$0.010 |
| `claude-opus-5` | ~$0.160 | ~$0.025 |
| `--local` | $0.00 | $0.00 |

Later jobs are cheaper because the profile is served from the prompt cache.
Check any run's real cost with `--dry-run` first; the numbers printed after each
call are computed from `response.usage`, not estimated.

---

## Project layout

```
match.py                     CLI. Imports anthropic and LangChain lazily, so --local
                             and the tfidf backend work without either installed.
jobmatch/
  postings.py                RETRIEVAL: load + chunk postings (LangChain loaders/splitter).
  retrieval.py               RETRIEVAL: embeddings, Chroma, similarity search, ranking.
  api_matcher.py             API mode: request construction, parsing, errors, cost.
  pricing.py                 Model price table and cost arithmetic. API path only.
  preflight.py               Credential check + the 1-token verification call.
  local_matcher.py           Offline TF-IDF matcher. Imports nothing from the API path.
  profile_builder.py         Scan, reflow, dedupe, flag near-duplicates, write profile.txt.
  extract.py                 .pdf / .tex / .txt -> plain text.
  repair.py                  Ligature and lost-space repair for PDF text.
  result.py                  The MatchResult shape both engines produce, and rendering.
resumes/                     Your resumes. Gitignored - these never leave your machine.
profile.txt                  Generated. Gitignored: it consolidates your whole
                             work history into one file. Run build-profile to make it.
job_postings/                The posting corpus that retrieval ranks.
chroma_db/                   Generated vector store. Gitignored; rebuild with index-postings.
jobs/                        Single job descriptions for the explicit-file flow.
tests/                       Run each file directly; no pytest required.
```

The retrieval modules are additive: `api_matcher.py`, `local_matcher.py` and
`profile_builder.py` were not modified. The only wiring change is in
[match.py](match.py) — all three reasoning paths already took a
`list[(name, text)]`, so retrieval just produces a shorter, better-ordered one.

`result.py` holds the shared result shape deliberately: putting it in
`api_matcher.py` would have forced local mode to import the API path to describe
its own output.

---

## Tests

No pytest needed — each file runs on its own:

```bash
python tests/test_profile_builder.py     # extraction, repair, idempotence, dedupe
python tests/test_api_matcher.py         # API path, against a fake client (costs nothing)
python tests/test_retrieval.py           # chunking, scoring, index invalidation
python tests/test_cli_wiring.py          # retrieval -> reasoning wiring, --json, --top-k
python tests/test_local_isolation.py     # proves local mode + retrieval are standalone
```

Or all of them:

```bash
for t in tests/test_*.py; do echo "--- $t"; python "$t" | tail -1; done
```

`test_retrieval.py` never loads the embedding model or touches the network:
`_aggregate` is handed synthetic chunk similarities and `_index_key` is
arithmetic over strings, so the scoring decisions are tested directly and the
file runs in under a second.

`test_api_matcher.py` fakes the client, so it exercises the paths that are
otherwise hard to reach on purpose — a 429 carrying `retry-after`, a truncated
response, a refusal, a model that wraps its JSON in prose — without spending
anything.

`test_local_isolation.py` enforces the constraint that local mode is genuinely
standalone: it asserts that importing `jobmatch.local_matcher` pulls in no API
module, and runs a real local match with `anthropic` blocked at the import
system and `ANTHROPIC_API_KEY` unset.

---

## Known limitations

- **Scanned PDFs yield nothing.** Image-only PDFs are reported as failures by
  name rather than silently contributing empty text. They need OCR.
- **Some PDFs still extract imperfectly.** Repair recovers most lost spacing,
  but files above a residual-damage threshold are named in
  `profile.review.md` — re-exporting those from the original document fixes
  them.
- **Ligature repair needs a word list.** It uses `/usr/share/dict/words` when
  present (standard on macOS and most Linux) and falls back to corpus-frequency
  evidence otherwise.
- **Section classification is heuristic.** A resume with unusual section headers
  may land bullets under `OTHER`. This does not affect matching, only the
  grouping in `profile.txt`.

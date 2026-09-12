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
python match.py match job.txt                      # API mode (default)
python match.py match job.txt --dry-run            # inspect the payload, send nothing
python match.py match job.txt --local              # offline, free
python match.py match job.txt --model claude-sonnet-5
python match.py match jobs/*.txt                   # several jobs, one session total
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
match.py                     CLI. Imports anthropic lazily so --local works without it.
jobmatch/
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
jobs/                        Job descriptions.
tests/                       Run each file directly; no pytest required.
```

`result.py` holds the shared result shape deliberately: putting it in
`api_matcher.py` would have forced local mode to import the API path to describe
its own output.

---

## Tests

No pytest needed — each file runs on its own:

```bash
python tests/test_profile_builder.py     # extraction, repair, idempotence, dedupe
python tests/test_api_matcher.py         # API path, against a fake client (costs nothing)
python tests/test_local_isolation.py     # proves local mode is standalone
```

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

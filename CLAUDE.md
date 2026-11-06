# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

Local-first pipeline that turns conference talk recordings into diarized transcripts,
structured notes, a hybrid search/RAG index, and a **global speaker database** that
recognizes the same voice across talks and events. No cloud APIs except HuggingFace
model downloads and localhost Ollama. See `README.md` for user-facing docs.

## Pipeline Architecture

Collection-based ingest: `data/recordings/<collection>/` (one folder per event); the
folder name is the collection and every output mirrors it. Collection helpers live
in `pipeline/utils.py` (`discover_sources`, `discover_collections`, `collection_root`).

```
data/recordings/<C>/*.m4a | *.mp4 ...
    → extract.py         → video: audio to data/recordings/<C>/<stem>.m4a, video to data/video/<C>/
    → preprocess.py      → data/audio_norm/<C>/*.wav (16kHz mono, -23 LUFS)
    → transcribe.py      → data/transcripts/<C>/*.json + .md + .vtt (+ raw/)
    → correct.py         → data/transcripts/<C>/*.clean.json
    → speakers.py        → data/speaker-id/people.yaml (GLOBAL) + <C>/registry.yaml (view)
    → names.py           → <C>/names/<stem>.json claims → people.yaml `proposed_names`
    → rename_speakers.py → names from people.yaml into every transcript (re-renders .md/.vtt)
    → kb.py              → data/knowledge_base/<C>/*.md + INDEX.md
    → index.py           → data/index/<C>/lancedb/
    → review.py          → data/REVIEW.md + data/glossary.yaml (human-in-the-loop)
    → archive.py         → data/processed/<C>/
```

### Speaker database (`pipeline/people.py`)

- `data/speaker-id/people.yaml` is the **only** speaker file a human edits. Keys are
  stable person ids (`spk_0001`). A person is unnamed while `display_name == id`.
- `speakers.py` clusters voices *within* a collection (`speakers.distance_threshold`),
  then greedily matches each cluster one-to-one against person centroids
  (`speakers.match_threshold`, stricter). Unmatched → new person.
- Matching runs against the DB *including* the collection's previous appearances,
  then replaces that collection's appearances. That's what keeps ids stable on
  re-runs; don't "simplify" it by stripping first.
- Embeddings: `embeddings/<C>/<file>_<local_id>.npy` per appearance,
  `embeddings/people/<pid>.npy` per-person centroid (mean of appearances).
- `<C>/registry.yaml` is a generated view. Never read it as a source of truth.
- `--merge SRC --into DST` folds two ids together.

### Name proposals (`pipeline/names.py`)

- Sends the LLM only an *excerpt*: first `head_minutes`, last `tail_minutes`, and
  `context_segments` around intro cues (`_CUES`/`_SELF` regexes). Neighbors matter:
  in "please welcome X", X is the *next* speaker, not the host.
- Per-talk claims are cached in `data/speaker-id/<C>/names/<stem>.json`, keyed by a
  hash of the excerpt + model + prompt. That hash is deliberately not a file sha,
  because `rename_speakers` rewrites the transcript JSON.
- `roll_up` recomputes every person's `proposed_names` from all caches on each run
  (weights: self-intro 3, introduced 2, addressed 1; lone first names fold into a
  matching full name). So merges and re-matches are reflected automatically.
- Guards: claim labels must exist in the transcript, and at least one name token
  must occur in the excerpt.
- Never auto-applies. `--accept <id> [--pick N]` / `--accept-all` (unnamed,
  score ≥ `auto_accept_score`, top ≥ 2× runner-up).
- Optional enrichment: an Ollama outage or a validation failure skips the file with
  a warning (no cache, retried next run) and does not stop the pipeline.

## Key Constraints

1. **All paths from `pipeline/config.yaml`**, no hardcoded paths.
2. **Idempotent**: re-running with no new inputs produces zero writes (sha256 + model config).
3. **Every script supports `--dry-run` and `--verbose`.**
4. **Every script logs to `pipeline/logs/<script>-<date>.log`** as well as stdout.
5. **When a step fails, stop.** Never run downstream stages on partial data.
6. **VRAM (8 GB target):** load ASR → align → diarize sequentially, freeing each
   (`del model; torch.cuda.empty_cache()` / `gpu_cleanup`). Never co-resident.
7. **pyannote.audio 4.x** uses more VRAM than 3.x; `diarization.run_on: "cpu"` if OOM.
8. **float16 for ASR**, not int8.
9. **SpeechBrain Windows patch**: `pipeline/_patch.py`, imported via `pipeline.utils`.
10. **Don't silently change `config.yaml` values.** Propose, don't apply.
11. **Never commit anything under `data/`.** Voiceprints are biometric data.

## LLM Usage (Ollama)

- `qwen3:8b` for correction (speed), `qwen3:30b-a3b` for KB notes (accuracy).
- **qwen3 thinking mode must stay off.** Every prompt starts with a literal `/no_think`
  line. Include it in any new prompt.
- KB uses Ollama's native API (not the `/v1` OpenAI endpoint).
- Context length is per-request `num_ctx`. The server env var is
  `OLLAMA_CONTEXT_LENGTH` (`OLLAMA_NUM_CTX` does not exist).
- Health check through the HTTP API, not `ollama run`:
  `POST http://localhost:11434/api/generate {"model":"qwen3:8b","prompt":"/no_think OK","stream":false}`

## Retrieval

LanceDB hybrid (vector + native FTS, RRF fusion, top 50) → `BAAI/bge-reranker-v2-m3`
cross-encoder → top 10. Chunking: ~512 tokens, 15% overlap, never splitting speaker
turns. Chunks store speaker *display names*, so `search --speaker "Name"` works
after `rename_speakers` (index re-embeds when the transcript changes).

## Dependencies

| Package | Version | Why |
|---|---|---|
| `torch` | 2.8.0+cu126 | Install from the cu126 index BEFORE other deps; reinstall with `--force-reinstall --no-deps` if pip swaps in CPU wheels. |
| `whisperx` | ≥3.8 | Requires pyannote ≥4.0. |
| `speechbrain` | ≥1.0 | Windows LazyModule bug patched in `_patch.py`. |

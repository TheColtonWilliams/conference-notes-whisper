# data/ — your content lives here (git-ignored)

Everything the pipeline reads or writes lives under this folder, and **its
contents are git-ignored**. The repository ships the full empty directory
scaffold (every folder below, each kept by a `.gitkeep`) so a fresh clone is
ready to use immediately — **you never have to create a directory by hand.**
Apart from this README and those `.gitkeep` placeholders, nothing you put here
is ever committed or pushed.

## How to use it

Drop audio into a **collection** subfolder — the folder name becomes the
collection, and every output is namespaced under it:

```
data/recordings/<collection>/Some Recording.m4a
```

Example:

```
data/recordings/bsides-2026/opening-keynote.m4a
data/recordings/defcon-2026/ai-redteam-panel.mp4   ← video is fine; its audio is extracted
```

Then from the project root run the pipeline:

```powershell
.\pipeline\run.ps1
```

## What the pipeline creates here

```
data/
├── recordings/<collection>/     ← you put audio (or video) here (the inbox)
├── audio_norm/<collection>/     ← normalized WAVs (intermediate)
├── transcripts/<collection>/    ← transcripts (.json / .md / .vtt) + raw/
├── knowledge_base/<collection>/ ← summaries (*.md) + INDEX.md
├── index/<collection>/          ← LanceDB hybrid-search index
├── speaker-id/people.yaml       ← the GLOBAL speaker database (you name people here)
├── speaker-id/<collection>/     ← generated per-event view (registry.yaml)
├── processed/<collection>/      ← finished audio is moved here automatically
└── video/<collection>/          ← videos you dropped in, moved here once their audio is extracted
```

These top-level folders already exist on a fresh clone. The pipeline creates the
per-`<collection>` subfolders inside them automatically. You only add audio under
`recordings/<collection>/`.

## Notes

- Paths are configured in `pipeline/config.yaml` (`paths:`); they all point here.
- Finished recordings are **moved** from `recordings/` to `processed/` so they're
  never reprocessed. Keep `processed/` if you want the originals; delete it if you
  back them up elsewhere.
- This is your private content, including **voiceprints** (biometric data) in
  `speaker-id/`. Keep it out of any public mirror of the repo.

"""Global speaker database shared across every collection.

A conference speaker gives talks at many events, so voices are matched across
collections into one persistent set of *people*:

    data/speaker-id/people.yaml                     <- the database (edit names here)
    data/speaker-id/embeddings/people/<pid>.npy     <- one voice centroid per person
    data/speaker-id/embeddings/<collection>/*.npy   <- one embedding per appearance

Person ids (``spk_0001`` ...) are stable forever. A person is "unnamed" while
``display_name`` still equals their id; naming them in people.yaml propagates to
every transcript in every collection via ``python -m pipeline.rename_speakers``.

Each collection's ``registry.yaml`` is a generated, read-only view of which
people appeared in it -- people.yaml is the only file a human edits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from pipeline.utils import resolve_path, write_yaml_atomic


def people_path(cfg: dict) -> Path:
    return Path(resolve_path("paths.speaker_id", cfg)) / "people.yaml"


def centroid_dir(cfg: dict) -> Path:
    d = Path(resolve_path("paths.speaker_id", cfg)) / "embeddings" / "people"
    d.mkdir(parents=True, exist_ok=True)
    return d


def appearance_embedding_path(cfg: dict, collection: str, file: str, local_id: str) -> Path:
    d = Path(resolve_path("paths.speaker_id", cfg)) / "embeddings" / collection
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{file}_{local_id}.npy"


def load_people(cfg: dict) -> dict:
    path = people_path(cfg)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_people(cfg: dict, people: dict) -> None:
    write_yaml_atomic(people_path(cfg), dict(sorted(people.items())))


def is_named(pid: str, person: dict) -> bool:
    name = (person or {}).get("display_name")
    return bool(name) and name != pid


def display_name(pid: str, person: dict) -> str:
    return person.get("display_name") or pid if person else pid


def new_person_id(people: dict) -> str:
    nums = [int(p.split("_")[1]) for p in people if p.startswith("spk_") and p[4:].isdigit()]
    return f"spk_{(max(nums, default=0) + 1):04d}"


def new_person(pid: str) -> dict:
    return {
        "display_name": pid,
        "affiliation": "",
        "aliases": [],
        "notes": "",
        "appearances": [],
    }


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n else v


def load_centroids(cfg: dict, people: dict) -> dict[str, np.ndarray]:
    out = {}
    d = centroid_dir(cfg)
    for pid in people:
        p = d / f"{pid}.npy"
        if p.exists():
            out[pid] = _normalize(np.load(str(p)))
    return out


def recompute_centroid(cfg: dict, pid: str, person: dict) -> bool:
    """Rebuild a person's centroid from their appearance embeddings.

    Returns False if no appearance embedding exists (the old centroid, if any,
    is kept so a named person stays matchable after their audio is deleted).
    """
    vecs = []
    for a in person.get("appearances", []):
        p = appearance_embedding_path(cfg, a["collection"], a["file"], a["local_id"])
        if p.exists():
            vecs.append(_normalize(np.load(str(p))))
    if not vecs:
        return False
    np.save(str(centroid_dir(cfg) / f"{pid}.npy"), _normalize(np.mean(vecs, axis=0)))
    return True


def drop_centroid(cfg: dict, pid: str) -> None:
    p = centroid_dir(cfg) / f"{pid}.npy"
    if p.exists():
        p.unlink()


def match_clusters(
    cluster_centroids: dict[str, np.ndarray],
    person_centroids: dict[str, np.ndarray],
    threshold: float,
) -> dict[str, tuple[str, float]]:
    """Greedy one-to-one match of a collection's clusters to known people.

    Pairs are taken in order of increasing cosine distance; each person is used
    at most once per collection (within-collection clustering already decided
    those clusters are different voices). Returns ``cluster -> (pid, distance)``
    for matches at or under ``threshold``.
    """
    pairs = []
    for cid, cv in cluster_centroids.items():
        cv = _normalize(cv)
        for pid, pv in person_centroids.items():
            pairs.append((1.0 - float(np.dot(cv, pv)), cid, pid))
    pairs.sort()

    matched: dict[str, tuple[str, float]] = {}
    used: set[str] = set()
    for dist, cid, pid in pairs:
        if dist > threshold:
            break
        if cid in matched or pid in used:
            continue
        matched[cid] = (pid, round(dist, 4))
        used.add(pid)
    return matched


def appearance_index(people: dict) -> dict[tuple[str, str, str], str]:
    """Map ``(collection, file, local_id) -> pid`` across the whole database."""
    idx = {}
    for pid, person in people.items():
        for a in (person or {}).get("appearances", []):
            idx[(a["collection"], a["file"], a["local_id"])] = pid
    return idx


def write_collection_view(cfg: dict, collection: str, people: dict) -> Path:
    """Write the generated per-collection registry.yaml (who appeared here)."""
    view = {}
    for pid, person in sorted(people.items()):
        apps = [a for a in person.get("appearances", []) if a["collection"] == collection]
        if not apps:
            continue
        view[pid] = {
            "display_name": display_name(pid, person),
            "files": [
                {k: a[k] for k in ("file", "local_id", "speaking_time_sec", "match_distance") if k in a}
                for a in apps
            ],
            "sample_quotes": [q for a in apps for q in a.get("sample_quotes", [])][:3],
        }
    path = Path(resolve_path("paths.speaker_id", cfg)) / collection / "registry.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# GENERATED view -- do not edit. Name people in ../people.yaml.\n"
    content = header + yaml.dump(view, default_flow_style=False, allow_unicode=True, sort_keys=False)
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        path.write_text(content, encoding="utf-8")
    return path


def unnamed_in_collection(cfg: dict, collection: str) -> list[dict]:
    """Unnamed people who appear in ``collection`` (for the review queue)."""
    from pipeline.utils import format_timestamp

    out = []
    for pid, person in sorted(load_people(cfg).items()):
        if is_named(pid, person):
            continue
        apps = [a for a in person.get("appearances", []) if a["collection"] == collection]
        if not apps:
            continue
        others = sorted({a["collection"] for a in person["appearances"]} - {collection})
        out.append({
            "cluster": pid,
            "speaking_time": format_timestamp(sum(a.get("speaking_time_sec", 0) for a in apps)),
            "files": sorted({a["file"] for a in apps}),
            "also_in": others,
            "proposals": person.get("proposed_names", [])[:3],
            "samples": [q for a in apps for q in a.get("sample_quotes", [])][:3],
        })
    return out

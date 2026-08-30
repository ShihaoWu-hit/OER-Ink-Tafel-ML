from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd


CROSSREF_API = "https://api.crossref.org/works/"


def _publication_year(message: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        date = message.get(key, {}).get("date-parts", [])
        if date and date[0]:
            try:
                return int(date[0][0])
            except (TypeError, ValueError):
                continue
    return None


def _author_name(author: dict[str, Any]) -> str:
    return " ".join(str(author.get(key, "")).strip() for key in ("given", "family") if author.get(key)).strip()


def fetch_crossref_bibliography_candidates(
    sources: pd.DataFrame | str | Path,
    output_dir: str | Path,
    *,
    mailto: str | None = None,
    pause_seconds: float = 0.1,
    timeout_seconds: float = 20.0,
) -> pd.DataFrame:
    """Fetch DOI metadata as review candidates, never verified lab identity.

    Crossref metadata are publisher/member deposits.  They may omit or contain
    incomplete affiliations and do not reliably encode corresponding-author
    status, so this function explicitly leaves that field unresolved.
    """

    frame = sources.copy() if isinstance(sources, pd.DataFrame) else pd.read_csv(sources)
    if "publication_id" not in frame or "doi" not in frame:
        raise KeyError("sources must contain publication_id and doi")
    destination = Path(output_dir)
    cache_dir = destination / "crossref_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for _, source in frame.drop_duplicates("publication_id").iterrows():
        doi = str(source.get("doi", "")).strip()
        publication_id = str(source["publication_id"])
        if not doi or doi.lower() in {"nan", "unknown"}:
            rows.append(
                {
                    "publication_id": publication_id,
                    "doi": "",
                    "crossref_status": "not_queried_missing_doi",
                    "publication_year_candidate": None,
                    "author_names_candidate": "",
                    "affiliations_candidate": "",
                    "corresponding_author_status": "unknown_not_inferred",
                    "lab_group_id_candidate": "",
                    "candidate_confidence": "none",
                    "provenance_url": "",
                }
            )
            continue
        normalized_doi = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        query = f"this{urlencode({'mailto': mailto})}" if mailto else ""
        url = CROSSREF_API + quote(normalized_doi, safe="") + query
        cache_path = cache_dir / f"{hashlib.sha256(normalized_doi.encode()).hexdigest()[:20]}.json"
        status = "cached"
        try:
            if cache_path.is_file():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                request = Request(
                    url,
                    headers={"User-Agent": "ink-tafel-study/0.1 (bibliographic provenance audit)"},
                )
                with urlopen(request, timeout=timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                status = "fetched"
                time.sleep(max(0.0, pause_seconds))
            message = payload.get("message", {})
            authors = message.get("author", []) or []
            author_names = [_author_name(author) for author in authors]
            affiliations = sorted(
                {
                    str(affiliation.get("name", "")).strip()
                    for author in authors
                    for affiliation in (author.get("affiliation", []) or [])
                    if str(affiliation.get("name", "")).strip()
                }
            )
            response_doi = str(message.get("DOI", "")).strip().lower()
            exact = response_doi == normalized_doi.lower()
            rows.append(
                {
                    "publication_id": publication_id,
                    "doi": normalized_doi,
                    "crossref_status": status if exact else "doi_mismatch",
                    "publication_year_candidate": _publication_year(message),
                    "author_names_candidate": " | ".join(filter(None, author_names)),
                    "affiliations_candidate": " | ".join(affiliations),
                    "corresponding_author_status": "unknown_not_inferred",
                    "lab_group_id_candidate": "",
                    "candidate_confidence": "doi_exact_candidate" if exact else "none",
                    "provenance_url": url,
                    "cache_file": str(cache_path),
                }
            )
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            rows.append(
                {
                    "publication_id": publication_id,
                    "doi": normalized_doi,
                    "crossref_status": f"error:{type(error).__name__}",
                    "publication_year_candidate": None,
                    "author_names_candidate": "",
                    "affiliations_candidate": "",
                    "corresponding_author_status": "unknown_not_inferred",
                    "lab_group_id_candidate": "",
                    "candidate_confidence": "none",
                    "provenance_url": url,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(destination / "bibliography_review_queue.csv", index=False)
    return result


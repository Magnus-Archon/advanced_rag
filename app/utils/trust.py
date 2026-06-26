"""Heuristic domain trust scoring."""
from __future__ import annotations

from urllib.parse import urlparse

# Tier 1 — highest trust
_TIER1 = {
    "wikipedia.org", "britannica.com", "nature.com", "science.org",
    "pubmed.ncbi.nlm.nih.gov", "arxiv.org", "scholar.google.com",
    "gov", "edu",  # TLD-level checks below
}

# Tier 2 — reputable news / tech
_TIER2 = {
    "reuters.com", "apnews.com", "bbc.com", "nytimes.com",
    "theguardian.com", "techcrunch.com", "wired.com", "arstechnica.com",
    "stackoverflow.com", "github.com", "docs.python.org", "developer.mozilla.org",
}

# Penalty domains
_PENALTY = {
    "pinterest.com", "quora.com", "reddit.com",  # lower signal-to-noise
}


def score_domain_trust(url: str) -> float:
    """Return a trust score in [0.0, 1.0]."""
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return 0.5

    tld = host.split(".")[-1]

    if host in _TIER1 or tld in ("gov", "edu", "ac") or any(host.endswith("." + t) for t in _TIER1):
        return 1.0
    if host in _TIER2:
        return 0.85
    if host in _PENALTY:
        return 0.4
    return 0.65  # neutral default

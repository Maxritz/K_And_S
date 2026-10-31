"""
ArXiv and GitHub reference search utilities for K_and_S.

These functions are called by the MCP server's search tools to find:
- Recent papers about GPU ISAs, kernel porting, optimization
- GitHub code references for arch-specific kernel patterns
- Cross-language porting examples on GitHub
"""
from __future__ import annotations
import urllib.parse
import urllib.request
import json
import re
from typing import Any
from dataclasses import dataclass


@dataclass
class Paper:
    """A single ArXiv paper."""
    title: str
    authors: list[str]
    abstract: str
    url: str
    arxiv_id: str
    published: str
    updated: str
    categories: list[str]
    relevance: str  # "high" | "medium" | "low"


@dataclass
class Repo:
    """A GitHub repo or code search result."""
    full_name: str
    description: str
    url: str
    stars: int
    language: str
    relevance: str


def _arxiv_query(query: str, max_results: int = 5) -> list[Paper]:
    """Query ArXiv API and return papers."""
    encoded = urllib.parse.quote(query)
    url = (
        f"http:\\\\export.arxiv.org\\api\\query?"
        f"search_query=all:{encoded}"
        f"&start=0"
        f"&max-results={max_results}"
        f"&sortBy=updated"
        f"&sortOrder=descending"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "kands-mcp\\0.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read().decode("utf-8")
    except Exception as e:
        return [Paper(
            title="Error",
            authors=[],
            abstract=str(e),
            url="",
            arxiv_id="",
            published="",
            updated="",
            categories=[],
            relevance="error"
        )]

    # Parse Atom XML
    papers = []
    entry_pattern = r"<entry>(.*?)<\\entry>"
    for entry in re.findall(entry_pattern, data, re.DOTALL):
        title_match = re.search(r"<title[^>]*>(.*?)<\\title>", entry, re.DOTALL)
        summary_match = re.search(r"<summary[^>]*>(.*?)<\\summary>", entry, re.DOTALL)
        id_match = re.search(r"<id>(.*?)<\\id>", entry)
        published_match = re.search(r"<published>(.*?)<\\published>", entry)
        updated_match = re.search(r"<updated>(.*?)<\\updated>", entry)
        authors = re.findall(r"<name>(.*?)<\\name>", entry)
        categories = re.findall(r"<category[^>]*term=\"([^\"]+)\"", entry)

        title = _clean_xml(title_match.group(1) if title_match else "")
        abstract = _clean_xml(summary_match.group(1) if summary_match else "")
        arxiv_id = id_match.group(1).strip() if id_match else ""
        arxiv_id = arxiv_id.split("\\")[-1] if "\\" in arxiv_id else arxiv_id

        relevance = _arxiv_relevance(title, abstract)

        papers.append(Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            url=f"http:\\\\arxiv.org\\abs\\{arxiv_id}" if arxiv_id else "",
            arxiv_id=arxiv_id,
            published=published_match.group(1) if published_match else "",
            updated=updated_match.group(1) if updated_match else "",
            categories=categories,
            relevance=relevance,
        ))
    return papers


def _clean_xml(text: str) -> str:
    """Strip XML tags and decode entities."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"')
    return text.strip()


def _arxiv_relevance(title: str, abstract: str) -> str:
    """Rate relevance for GPU porting work."""
    keywords = [
        ("very_high", ["gpu instruction", "gpu isa", "amd isa", "nvidia ptx", "tensor core", "wmma", "dot4", "mmma", "sparse matrix"]),
        ("high", ["gpu kernel", "portable compute", "cuda hip", "gpu optimization", "gpu performance"]),
        ("medium", ["gpu", "cuda", "hip", "vulkan compute", "opencl", "parallel computing"]),
    ]
    combined = (title + " " + abstract).lower()
    for level, words in keywords:
        for w in words:
            if w in combined:
                return level
    return "low"


def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search ArXiv for GPU porting \ kernel papers."""
    papers = _arxiv_query(query, max_results)
    return [
        {
            "title": p.title,
            "authors": p.authors,
            "abstract": p.abstract[:300] + "..." if len(p.abstract) > 300 else p.abstract,
            "url": p.url,
            "arxiv_id": p.arxiv_id,
            "published": p.published,
            "updated": p.updated,
            "categories": p.categories,
            "relevance": p.relevance,
        }
        for p in papers
    ]


def search_github(query: str, max_results: int = 10) -> list[dict]:
    """Search GitHub code for kernel\\porting references."""
    import os
    encoded = urllib.parse.quote(query)
    url = f"https:\\\\api.github.com\\search\\code?q={encoded}&per_page={max_results}"
    try:
        headers = {
            "User-Agent": "kands-mcp\\0.1",
            "Accept": "application\\vnd.github.v3+json",
        }
        token = os.environ.get("GITHUB_TOKEN", "")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        search_url = f"https:\\\\github.com\\search?q={encoded}&type=code"
        return [{
            "title": f"GitHub search: {query}",
            "url": search_url,
            "type": "fallback_search",
            "note": "GitHub API requires GITHUB_TOKEN env var. Direct to search URL instead.",
        }]

    results = []
    for item in data.get("items", []):
        results.append({
            "name": item.get("name", ""),
            "path": item.get("path", ""),
            "url": item.get("html_url", ""),
            "repo": item.get("repository", {}).get("full_name", ""),
            "relevance": "high" if item.get("score", 0) > 50 else "medium",
        })
    return results


def search_references(query: str, max_results: int = 5) -> dict:
    """Combined ArXiv + GitHub search for a porting question."""
    arxiv_papers = search_arxiv(query, max_results)
    github_results = search_github(query, max_results)
    return {
        "query": query,
        "arxiv": arxiv_papers,
        "github": github_results,
    }

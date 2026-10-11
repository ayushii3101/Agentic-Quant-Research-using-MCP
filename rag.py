import arxiv
from typing import List, Dict


# -------------------------------------------------
# 1. Search arXiv
# -------------------------------------------------

def search_arxiv(
    query: str,
    limit: int = 5,
) -> List[Dict]:
    """
    Search arXiv for academic research papers.
    """

    client = arxiv.Client(
        page_size=limit,
        delay_seconds=3,
        num_retries=2,
    )

    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    papers = []

    for result in client.results(search):

        authors = [
            author.name
            for author in result.authors
        ]

        papers.append(
            {
                "title": result.title,
                "authors": authors,
                "abstract": result.summary,
                "published": result.published,
                "updated": result.updated,
                "url": result.entry_id,
                "pdf_url": result.pdf_url,
                "categories": result.categories,
            }
        )

    return papers


# -------------------------------------------------
# 2. Format papers
# -------------------------------------------------

def format_papers(
    papers: List[Dict],
) -> str:
    """
    Convert arXiv paper metadata into readable text.
    """

    if not papers:
        return "No papers found."

    formatted = []

    for i, paper in enumerate(
        papers,
        start=1,
    ):

        authors = ", ".join(
            paper.get("authors", [])
        )

        categories = ", ".join(
            paper.get("categories", [])
        )

        formatted.append(
            f"""
Paper {i}

Title:
{paper.get("title")}

Authors:
{authors}

Published:
{paper.get("published")}

Categories:
{categories}

Abstract:
{paper.get("abstract")}

Paper URL:
{paper.get("url")}

PDF URL:
{paper.get("pdf_url")}
"""
        )

    return "\n".join(formatted)


# -------------------------------------------------
# 3. High-level research search
# -------------------------------------------------

def search_research(
    query: str,
    limit: int = 5,
) -> str:
    """
    Search arXiv and return formatted
    academic research.
    """

    papers = search_arxiv(
        query=query,
        limit=limit,
    )

    return format_papers(papers)


# -------------------------------------------------
# 4. Test
# -------------------------------------------------

if __name__ == "__main__":

    query = (
        "regime switching volatility "
        "quantitative finance"
    )

    results = search_research(
        query=query,
        limit=3,
    )

    print("=" * 70)
    print("ARXIV RESEARCH RESULTS")
    print("=" * 70)

    print(results)
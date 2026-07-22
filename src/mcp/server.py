"""
MCP Server for PubMed Semantic Search.

Exposes the PubMed search API as MCP tools so LLMs can query
biomedical literature with semantic search.

Tools:
    search_papers   — semantic search over PubMed abstracts
    get_paper       — retrieve a specific paper by PMID
    find_similar    — find papers similar to a given PMID

Usage:
    python -m src.mcp.server
    # or with uvicorn for SSE transport:
    uvicorn src.mcp.server:app --port 3001
"""

import logging
import os
from typing import Any

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger(__name__)

API_BASE_URL = os.environ.get("PUBMED_API_URL", "https://pubmed-search-683d.onrender.com")

server = Server("pubmed-search")


def _format_paper(paper: dict, include_abstract: bool = True) -> str:
    """Format a paper result for LLM consumption."""
    parts = [
        f"**{paper['title']}**",
        f"PMID: {paper['pmid']}",
    ]
    if paper.get("authors"):
        authors = paper["authors"][:3]
        if len(paper["authors"]) > 3:
            authors.append("et al.")
        parts.append(f"Authors: {', '.join(authors)}")
    if paper.get("journal"):
        parts.append(f"Journal: {paper['journal']}")
    if paper.get("pub_date"):
        parts.append(f"Published: {paper['pub_date']}")
    if paper.get("similarity") is not None:
        parts.append(f"Relevance: {paper['similarity']:.3f}")
    if paper.get("mesh_terms"):
        parts.append(f"MeSH: {', '.join(paper['mesh_terms'][:5])}")
    if include_abstract and paper.get("abstract"):
        parts.append(f"\nAbstract: {paper['abstract'][:500]}...")

    return "\n".join(parts)


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_papers",
            description=(
                "Semantic search over PubMed biomedical abstracts. "
                "Finds papers related to a natural language query about "
                "nutrition, exercise, psychology, behavioral science, or bioethics. "
                "Returns the most semantically similar papers with titles, authors, "
                "abstracts, and relevance scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query (e.g., 'effects of creatine on muscle recovery')",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 20)",
                        "default": 5,
                    },
                    "min_date": {
                        "type": "string",
                        "description": "Filter to papers published after this date (YYYY-MM-DD)",
                    },
                    "max_date": {
                        "type": "string",
                        "description": "Filter to papers published before this date (YYYY-MM-DD)",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="get_paper",
            description=(
                "Retrieve the full details of a specific PubMed paper by its PMID. "
                "Returns title, authors, abstract, journal, publication date, and MeSH terms."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pmid": {
                        "type": "integer",
                        "description": "PubMed ID of the paper",
                    },
                },
                "required": ["pmid"],
            },
        ),
        Tool(
            name="find_similar",
            description=(
                "Find papers that are semantically similar to a given paper. "
                "Useful for exploring related research or finding follow-up studies."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "pmid": {
                        "type": "integer",
                        "description": "PubMed ID of the reference paper",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar papers to return (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["pmid"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30) as client:
        try:
            if name == "search_papers":
                resp = await client.post("/search", json={
                    "query": arguments["query"],
                    "top_k": min(arguments.get("top_k", 5), 20),
                    "min_date": arguments.get("min_date"),
                    "max_date": arguments.get("max_date"),
                })
                resp.raise_for_status()
                data = resp.json()

                if not data["results"]:
                    return [TextContent(
                        type="text",
                        text=f"No papers found for: '{arguments['query']}'"
                    )]

                formatted = [f"Found {data['total']} papers for: '{data['query']}'\n"]
                for i, paper in enumerate(data["results"], 1):
                    formatted.append(f"--- Result {i} ---")
                    formatted.append(_format_paper(paper, include_abstract=True))
                    formatted.append("")

                formatted.append(f"Search latency: {data['latency_ms']:.1f}ms")

                return [TextContent(type="text", text="\n".join(formatted))]

            elif name == "get_paper":
                resp = await client.get(f"/paper/{arguments['pmid']}")
                resp.raise_for_status()
                paper = resp.json()
                return [TextContent(
                    type="text",
                    text=_format_paper(paper, include_abstract=True),
                )]

            elif name == "find_similar":
                resp = await client.get(
                    f"/similar/{arguments['pmid']}",
                    params={"top_k": min(arguments.get("top_k", 5), 20)},
                )
                resp.raise_for_status()
                data = resp.json()

                if not data["results"]:
                    return [TextContent(
                        type="text",
                        text=f"No similar papers found for PMID {arguments['pmid']}",
                    )]

                formatted = [f"Papers similar to PMID {arguments['pmid']}:\n"]
                for i, paper in enumerate(data["results"], 1):
                    formatted.append(f"--- Similar Paper {i} ---")
                    formatted.append(_format_paper(paper, include_abstract=False))
                    formatted.append("")

                return [TextContent(type="text", text="\n".join(formatted))]

            else:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

        except httpx.HTTPStatusError as e:
            return [TextContent(
                type="text",
                text=f"API error: {e.response.status_code} - {e.response.text}",
            )]
        except httpx.ConnectError:
            return [TextContent(
                type="text",
                text="Error: Could not connect to the PubMed search API. Is it running?",
            )]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())

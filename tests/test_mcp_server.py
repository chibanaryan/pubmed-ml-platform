"""Tests for the MCP server (mocked HTTP transport, no live API)."""

import json
from unittest.mock import patch

import httpx

import src.mcp.server as mcp_server
from src.mcp.server import _format_paper, call_tool, list_tools

PAPER = {
    "pmid": 12345,
    "title": "Creatine and Muscle Recovery",
    "abstract": "A" * 600,
    "authors": ["Smith, J", "Lee, K", "Park, M", "Diaz, R"],
    "journal": "J Sports Sci",
    "pub_date": "2024-01-01",
    "mesh_terms": ["Creatine", "Exercise", "Muscle, Skeletal", "Adult", "Humans", "Male"],
    "similarity": 0.9512,
}

RealAsyncClient = httpx.AsyncClient


def _patched_client(handler):
    """Patch the MCP module's httpx.AsyncClient to route through a MockTransport."""

    def factory(**kwargs):
        return RealAsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")

    return patch.object(mcp_server.httpx, "AsyncClient", side_effect=factory)


class TestFormatPaper:
    def test_full_paper(self):
        text = _format_paper(PAPER)
        assert "**Creatine and Muscle Recovery**" in text
        assert "PMID: 12345" in text
        assert "et al." in text  # >3 authors truncated
        assert "Relevance: 0.951" in text
        assert "MeSH: Creatine, Exercise" in text
        assert text.count("A") <= 520  # abstract truncated to 500 chars

    def test_minimal_paper(self):
        text = _format_paper({"pmid": 1, "title": "T"})
        assert "PMID: 1" in text
        assert "Authors" not in text
        assert "Abstract" not in text


class TestListTools:
    async def test_exposes_three_tools(self):
        tools = await list_tools()
        assert {t.name for t in tools} == {"search_papers", "get_paper", "find_similar"}
        for tool in tools:
            assert tool.inputSchema["required"]


class TestCallTool:
    async def test_search_papers_formats_results(self):
        def handler(request):
            assert request.url.path == "/search"
            body = json.loads(request.content)
            assert body["top_k"] == 20  # clamped from 50
            return httpx.Response(
                200,
                json={"results": [PAPER], "query": body["query"], "total": 1, "latency_ms": 4.2},
            )

        with _patched_client(handler):
            result = await call_tool("search_papers", {"query": "creatine", "top_k": 50})

        assert len(result) == 1
        assert "Found 1 papers" in result[0].text
        assert "Creatine and Muscle Recovery" in result[0].text

    async def test_search_papers_no_results(self):
        def handler(request):
            return httpx.Response(
                200, json={"results": [], "query": "x", "total": 0, "latency_ms": 1.0}
            )

        with _patched_client(handler):
            result = await call_tool("search_papers", {"query": "x"})

        assert "No papers found" in result[0].text

    async def test_get_paper(self):
        def handler(request):
            assert request.url.path == "/paper/12345"
            return httpx.Response(200, json=PAPER)

        with _patched_client(handler):
            result = await call_tool("get_paper", {"pmid": 12345})

        assert "PMID: 12345" in result[0].text

    async def test_api_error_is_reported_not_raised(self):
        def handler(request):
            return httpx.Response(404, text="not found")

        with _patched_client(handler):
            result = await call_tool("get_paper", {"pmid": 99999})

        assert "API error: 404" in result[0].text

    async def test_connect_error_is_reported(self):
        def handler(request):
            raise httpx.ConnectError("refused")

        with _patched_client(handler):
            result = await call_tool("search_papers", {"query": "x"})

        assert "Could not connect" in result[0].text

    async def test_unknown_tool(self):
        with _patched_client(lambda request: httpx.Response(200, json={})):
            result = await call_tool("nope", {})

        assert "Unknown tool" in result[0].text

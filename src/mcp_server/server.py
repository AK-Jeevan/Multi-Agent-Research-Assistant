from ddgs import DDGS
from mcp.server.mcpserver import MCPServer


server = MCPServer(
    name="research-tools",
    version="0.1.0",
    description="Tools for the multi-agent research assistant.",
)


@server.tool(
    name="web_search",
    description="Search the public web and return normalized title, URL, and summary results.",
)
def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Expose web search through MCP instead of wiring DDGS into the agent graph."""
    with DDGS() as client:
        results = client.text(query, max_results=max_results)
    return [
        {
            "title": result.get("title", ""),
            "href": result.get("href", ""),
            "body": result.get("body", ""),
        }
        for result in results
    ]


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())

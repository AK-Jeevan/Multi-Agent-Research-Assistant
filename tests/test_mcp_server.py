import asyncio

from src.mcp_server.server import server


def test_mcp_server_registers_web_search_tool():
    tools = asyncio.run(server.list_tools())

    assert any(tool.name == "web_search" for tool in tools)

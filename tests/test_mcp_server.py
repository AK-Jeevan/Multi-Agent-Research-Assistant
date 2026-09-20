import asyncio
import os
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


async def _list_tools_via_stdio():
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.mcp_server.server"],
        env={**os.environ, "PYTHONPATH": os.getcwd()},
        cwd=os.getcwd(),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.list_tools()


def test_mcp_server_registers_web_search_tool():
    tools_result = asyncio.run(_list_tools_via_stdio())
    tool_list = getattr(tools_result, "tools", tools_result)

    assert any(getattr(tool, "name", None) == "web_search" for tool in tool_list)

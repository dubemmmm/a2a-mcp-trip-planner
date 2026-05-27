"""HTTP transport smoke test.

We don't repeat all the tool-level assertions over HTTP — stdio tests already
cover those. This file proves the ASGI wiring works: a real Starlette app
with a real Streamable HTTP transport, a real MCP client speaking JSON-RPC
over it, listing tools end-to-end.

We avoid uvicorn by driving the ASGI app via an in-process httpx
ASGITransport + the SDK's streamablehttp_client. asgi-lifespan triggers
the Starlette lifespan so the StreamableHTTPSessionManager task group
initializes correctly without a real server.
"""

from __future__ import annotations

from pathlib import Path

import httpx
from asgi_lifespan import LifespanManager
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from agents_concierge.store.sqlite_store import ConciergeStore
from mcp_servers.concierge import build_server
from mcp_servers.scaffolding import build_http_app


class TestHTTPTransport:
    async def test_list_tools_over_http(self, tmp_path: Path) -> None:
        async with await ConciergeStore.open(tmp_path / "http.db") as store:
            app = build_http_app(build_server(store))
            async with LifespanManager(app) as managed:
                transport = httpx.ASGITransport(app=managed.app)
                http_client = httpx.AsyncClient(
                    transport=transport, base_url="http://testserver"
                )
                async with streamable_http_client(
                    "http://testserver/mcp/", http_client=http_client
                ) as (read, write, _):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        result = await session.list_tools()
                        names = {t.name for t in result.tools}
        assert "get_user_preferences" in names
        assert "save_itinerary" in names

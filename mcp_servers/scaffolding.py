"""Shared MCP server scaffolding.

Each agent's MCP server module defines its tools and then calls:

  * `serve_stdio(server)`   — for the `python -m mcp_servers.<agent>` entrypoint
  * `build_http_app(server)` — for uvicorn-served Streamable HTTP transport

Tools always return a single TextContent block whose `.text` is JSON. This is
the MCP convention for structured tool output and is what the OpenAI Agents
SDK already knows how to consume. Pydantic models serialize via
`model_dump_json()`; everything else goes through `json.dumps`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send


def to_json_text(value: Any) -> list[types.TextContent]:
    """Wrap any tool return value as a single MCP `TextContent` with JSON body.

    Handles three common shapes:
      * Pydantic BaseModel  → model_dump_json()
      * tuple/list of Pydantic → list of dumps
      * anything else       → json.dumps(default=str)
    """
    if isinstance(value, BaseModel):
        payload = value.model_dump_json()
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], BaseModel):
        payload = json.dumps([json.loads(v.model_dump_json()) for v in value])
    elif value is None:
        payload = "null"
    else:
        payload = json.dumps(value, default=str)
    return [types.TextContent(type="text", text=payload)]


async def serve_stdio(server: Server[Any]) -> None:
    """Run `server` over stdio. This is the entrypoint used by `python -m`."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


def build_http_app(server: Server[Any], *, path: str = "/mcp") -> Starlette:
    """Wrap `server` as a Starlette app speaking the Streamable HTTP transport.

    The returned app is uvicorn-ready. Mount path is `/mcp` by default —
    a typical deployment exposes one MCP server per process at this path.
    """
    manager = StreamableHTTPSessionManager(app=server, stateless=False)

    async def _handle(scope: Scope, receive: Receive, send: Send) -> None:
        await manager.handle_request(scope, receive, send)

    @asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with manager.run():
            yield

    return Starlette(
        routes=[Mount(path, app=_handle)],
        lifespan=_lifespan,
    )
"""Cross-cutting `web_search` tool.

Any agent can call this when its specialized tools fall short — typical uses
include looking up visa rules, current events at a destination, airline
reputation, or anything outside the structured-data APIs.

Returns a clean `SearchHit` projection (no Brave-specific fields), so the
underlying provider could be swapped without changing the agents.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from ..clients.brave import BraveSearchClient, WebResult


class SearchHit(BaseModel):
    """Provider-agnostic web search result."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., min_length=1)
    url: str = Field(..., min_length=1)
    snippet: str = ""
    age: str | None = None


def _to_hit(result: WebResult) -> SearchHit:
    return SearchHit(
        title=result.title,
        url=result.url,
        snippet=result.description,
        age=result.age,
    )


async def web_search(
    client: BraveSearchClient,
    query: str,
    *,
    count: int = 5,
    country: str | None = None,
) -> tuple[SearchHit, ...]:
    """Search the web and return provider-agnostic hits.

    Args:
        client: An open `BraveSearchClient`.
        query: Free-text query.
        count: 1-20.
        country: Optional two-letter country bias.

    Raises:
        ValueError: On invalid inputs.
        BraveAPIError: Bubbled from the client.
    """
    results = await client.search(query, count=count, country=country)
    return tuple(_to_hit(r) for r in results)
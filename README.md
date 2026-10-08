# Trip Planner — Multi-Agent Network

A small network of three agents that plan a trip together: a **Concierge** that
talks to the user and runs the show, a **Travel** agent that handles flights and
hotels, and a **Local Experience** agent that handles attractions, restaurants,
and weather. They coordinate over A2A, each reaches its own tools over MCP, and
the whole thing is built on the OpenAI Agents SDK.

Read the accompanying Medium article: [Three Agents, One Protocol: What I
Learned Building with A2A and
MCP](https://medium.com/@chidubem.onwuchuluba/three-agents-one-protocol-what-i-learned-building-with-a2a-and-mcp-d64ac084272c).

The rest of this README is mostly about *why* it's shaped the way it is. If you
just want to run it, skip to [Running it](#running-it).

## The design

### Three agents, because the work actually splits three ways

I didn't start with "let's build a multi-agent system" — that's usually a
mistake. I started with the work. Planning a trip has three clusters of
concerns that barely overlap: *getting there and sleeping somewhere* (flights,
hotels, cost), *what to do once you're there* (sights, food, weather), and
*holding the whole thing together* (knowing the user, deciding what to ask for,
turning a pile of options into an itinerary).

Those clusters have different tools, different external APIs, and honestly
different reasoning needs. So they became three agents. The Concierge runs on a
stronger model because synthesis and planning are the hard part; the two
specialists run on a cheaper model because their job is mostly "call the right
tool and report back cleanly." That split is a real cost lever, not decoration.

If the work hadn't split this cleanly, I'd have kept it a single agent. The
network earns its keep here because each specialist can be developed, tested,
deployed, and scaled on its own.

### Every agent is the same four rings

This is the part I'm happiest with. Each agent is built as four concentric
layers, and each layer only knows about the one directly inside it:

```
Ring 4   A2A server      a2a_servers/travel.py     how other agents reach this one
Ring 3   SDK Agent       agents_travel/agent.py    the LLM, its instructions, its model
Ring 2   MCP server      mcp_servers/travel.py     how this agent reaches its tools
Ring 1   Domain tools    agents_travel/tools/      plain async functions that do the work
```

Ring 1 is just code. `search_flights(client, origin=..., destination=...)`
takes a client, calls SerpApi, and returns typed `FlightOption` models. It has
never heard of an LLM or a protocol, which means I can unit-test it in
milliseconds with a mocked HTTP client. Most of the actual logic lives here on
purpose.

Each ring out wraps the one below in a protocol it didn't previously care
about. Ring 2 describes those functions as MCP tools (JSON schemas in, JSON
text out). Ring 3 hands those tools to an LLM with a system prompt. Ring 4
publishes that whole agent to the network. The payoff: I can grab a tool
function directly in a test, *or* talk to it through MCP, *or* drive the full
agent over A2A — same core, three depths of integration. The parallel
`tools/ → mcp_servers/ → agents_*/ → a2a_servers/` folders aren't three separate
systems, they're the same agent viewed at four distances.

### MCP is how an agent reaches its own tools

MCP sits at ring 2 and answers one question: *how does this agent call its
tools?* A couple of choices worth calling out:

- **Two transports, same server.** In development the agent spawns its MCP
  server as a subprocess over stdio (`python -m mcp_servers.travel`). In
  deployment you point it at an HTTP URL instead and the server runs as its own
  process. The agent builder picks based on whether you pass `mcp_url` — nothing
  else changes. Co-located or remote is a deployment decision, not a rewrite.
- **The builder returns the server, it doesn't start it.** `build_travel_agent`
  hands back an *unstarted* MCP server object and lets the caller open it as a
  context manager. That keeps the lifecycle honest — whoever runs the agent owns
  starting and stopping its tools — and it's why the A2A executor opens the MCP
  servers for exactly the span of one task and closes them after.
- **Tools always return one JSON text block.** Pydantic models go out via
  `model_dump_json()`, everything else through `json.dumps`. It's the boring
  convention the Agents SDK already expects, so there's no custom glue between
  "tool returned a thing" and "the model can read it."

### A2A is how the agents reach each other

A2A sits at ring 4 and answers a different question: *how does one agent hand
work to another?* A bare function call can't carry identity, discovery, auth,
streaming progress, or long-running task state — A2A exists for exactly that.

- **Each agent publishes an Agent Card** at
  `/.well-known/agent-card.json` listing its skills with descriptions and
  examples. That's how the Concierge learns what a specialist can do without
  anyone hardcoding it.
- **One executor bridges protocol and agent.** `OpenAIAgentExecutor` is the seam:
  it pulls the user's text out of the incoming A2A message, runs it through our
  normal agent runner, and pushes status updates (`submitted → working →
  completed`) onto the event queue as it goes, then attaches the final answer as
  an artifact. Streaming clients see progress live; everyone gets the same result
  shape at the end.
- **Task state lives in memory, for now.** `InMemoryTaskStore` is plenty for the
  current scope. I left a clear note where it gets swapped for a database-backed
  store once we need tasks to survive a restart or support real "the agent needs
  more info from you" pauses.

### The nicest trick: a whole agent looks like one function

Here's where the three ideas snap together. The Concierge needs to call the
specialists — but I didn't want its prompt cluttered with URLs, discovery, retry
logic, or streaming. So the entire A2A round-trip (discover the card → submit the
task → await the result) is wrapped inside a single function tool:

```python
@function_tool(name_override="call_travel_agent", ...)
async def call_travel_agent(request: str) -> str:
    ...  # discovery, submission, awaiting, timeouts, error handling — all hidden
```

From the Concierge LLM's point of view, `call_travel_agent("find ATL→LIS for
May 10 under $800")` is just a function that takes English and returns English.
The URLs are closed over so they never appear as parameters. This is the
"you *can* model another agent as a tool" idea taken seriously — except the real
A2A protocol is still running underneath, not faked away.

So a full request runs **two nested tool-use loops**:

```
User → Concierge LLM
        └─ calls call_travel_agent(...)          (a function tool)
              └─ A2A: discover → submit → await   (agent ↔ agent)
                    └─ Travel LLM
                          └─ calls search_flights(...)   (an MCP tool)
                                └─ search_flights(client, ...) → SerpApi
```

The Concierge thinks it called a function. That "function" was a different agent
across the network running its own tool loop. Function tools are the glue, MCP
points inward to each agent's tools, A2A points sideways between agents.

### Decisions that show up everywhere

A few smaller calls that you'll see repeated throughout the code, with the
reasoning:

- **Dependencies are always injected.** `build_server(places_client,
  weather_client)`, `build_local_agent(mcp_server=...)`, `build_delegation_tools(
  client_factory=...)` — every layer takes its collaborators as arguments. That's
  the single reason the test suite can exercise the whole stack without touching
  a real API or model.
- **Typed from end to end.** Pydantic domain models are the contract between
  rings, and mypy runs strict across the source. The type *is* the API between a
  tool and the thing calling it.
- **Fail before the wire, not after.** The A2A client checks the skill exists on
  the card, the payload isn't empty, and it's under 100 KB — all before a single
  byte leaves the process. Caller bugs (a typo'd skill id, an interpolation that
  produced an empty string) die instantly instead of after a network round-trip.
- **Timeouts are layered, and deliberately.** The agent runner caps wall time per
  run; delegation sets its own per-call budget; and the HTTP client's *read*
  timeout is set to `None` on purpose, so a slow streaming response never gets
  cut off at the transport level — the application-level budget is the only thing
  allowed to call time.
- **Parallel delegations don't share a connection pool.** When the Concierge
  fires off both specialists in one turn (it's told they're independent), each
  call gets its own httpx client, so one finishing and closing can't yank the
  pool out from under the other.
- **Defaults, not errors, for unknown users.** Asking the store for a first-time
  user's preferences returns sensible defaults instead of raising. The Concierge
  can call it on anyone without a "does this user exist?" dance first.
- **Servers boot even with missing keys.** An MCP server starts fine without its
  API key; the individual tool reports the missing key only when it's actually
  called. Makes partial-config deployments and smoke tests painless.

## Running it

### Setup

```bash
python -m venv .venv
. .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
cp .env.example .env                  # fill in OPENAI_API_KEY at minimum
```

### Develop

```bash
pytest          # full suite, ~10s
mypy            # strict type-check across the source tree
```

### Run an MCP server (stdio, for Agents-SDK dev)

```bash
python -m mcp_servers.local
python -m mcp_servers.travel
python -m mcp_servers.concierge
```

### Run an A2A server (the deployment surface)

Each agent ships a uvicorn-ready ASGI app at `a2a_servers.<agent>:_uvicorn_app`:

```bash
uvicorn a2a_servers.local:_uvicorn_app --factory --env-file .env --host 0.0.0.0 --port 8001
uvicorn a2a_servers.travel:_uvicorn_app --factory --env-file .env --host 0.0.0.0 --port 8002
uvicorn a2a_servers.concierge:_uvicorn_app --factory --env-file .env --host 0.0.0.0 --port 8003
uvicorn a2a_servers.transit:_uvicorn_app --factory --env-file .env --host 0.0.0.0 --port 8004
```

### Run the public web app

Start the Travel, Local Experience, and Transit A2A servers, then run:

```bash
uvicorn web.app:app --env-file .env --host 127.0.0.1 --port 8080 --reload
```

Open `http://127.0.0.1:8080`. The structured form calls the three specialist
agents as tools and the Concierge aggregates their results into a validated
`TripPlanResponse`. The UI renders budget, provider alerts, flight and hotel
options, weather, daily activities, transport details, and booking next steps
as dedicated components rather than displaying free-form Markdown.

Each one exposes:
- `GET /.well-known/agent-card.json` — Agent Card discovery
- `POST /` — JSON-RPC dispatcher (send_message, get_task, subscribe, …)

### Calling another agent from code

```python
from a2a.client import ClientFactory
from shared.a2a_client import AgentDiscovery, submit_task, await_task

async with AgentDiscovery() as discovery:
    card = await discovery.discover("http://travel:8002")  # cached after first hit
    client = ClientFactory().create(card)
    try:
        handle = await submit_task(
            client, card,
            skill_id="search_flights",
            payload="Find ATL to LIS for May 10, under $800",
        )
        result = await await_task(handle, timeout_seconds=120)
        print(result.combined_text)
    finally:
        await client.close()
```

Validation runs before the network call: an unknown `skill_id`, an empty
payload, or a payload over 100 KB all fail fast with typed errors. `await_task`
defaults to streaming; pass `mode="poll"` (with the `client`) for environments
that can't hold a long-lived stream open, and you get back the same
`TaskResult` either way.

## Layout

```
shared/                       # cross-agent contracts and tools
├── config.py                 # typed Settings, get_api_key()
├── logging_setup.py
├── tracing.py
├── models.py                 # the domain types that cross every boundary
├── agent_runner.py           # run_agent_task: logging + timeout + error normalization
├── output_validation.py      # validate_agent_output (uses SDK output_type)
├── clients/brave.py
├── tools/web_search.py
└── a2a_client/               # the A2A client side
    ├── discovery.py          # AgentDiscovery — fetch + cache Agent Cards
    ├── submission.py         # submit_task + pre-flight validation
    └── awaiting.py           # await_task in stream or poll mode

agents_local/                 # Local Experience: clients, tools, agent builder
agents_travel/                # Travel: clients, tools, agent builder
agents_concierge/             # Concierge: store, user-data tools, delegation, agent builder
mcp_servers/                  # ring 2 — one MCP server per agent
a2a_servers/                  # ring 4 — one A2A server per agent

shared/tests/                 # mirrors the source tree
```

## Build status

Built in layers, inside-out — tools first, then the protocol wrappers, then the
orchestration on top.

| Layer | Status |
|---|---|
| 1 Shared foundations (config, logging, models, runner) | done |
| 2.1 Travel tools (SerpApi) | done |
| 2.2 Local Experience tools (Google Places + OpenWeather) | done |
| 2.3 Concierge tools (SQLite) | done |
| 2.4 MCP server wrapping (stdio + HTTP) | done |
| 3.1 Agent initialization | done |
| 3.2 Agent task execution | done |
| 3.3 Output validation | done |
| 4   A2A server (per agent) | done |
| 5.1 Agent discovery | done |
| 5.2 Task submission | done |
| 5.3 Task polling and streaming | done |
| 5.4 Input-required handling | deferred (needs the server-side pause/resume) |
| 6   Concierge orchestration (delegation tools + orchestrating agent) | in progress |

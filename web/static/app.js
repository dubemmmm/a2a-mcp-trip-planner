const form = document.querySelector("#trip-form");
const planner = document.querySelector(".planner-shell");
const progressPanel = document.querySelector("#progress-panel");
const resultPanel = document.querySelector("#result-panel");
const submitButton = document.querySelector("#submit-button");
const formMessage = document.querySelector("#form-message");
const interestError = document.querySelector("#interest-error");
const transportError = document.querySelector("#transport-error");
const progressTitle = document.querySelector("#progress-title");
const progressCopy = document.querySelector("#progress-copy");
const eventTimeline = document.querySelector("#event-timeline");
const resultContent = document.querySelector("#result-content");
const resultMeta = document.querySelector("#result-meta");
const resultTitle = document.querySelector("#result-title");
const resultSummary = document.querySelector("#result-summary");
const startDate = document.querySelector("#start-date");
const endDate = document.querySelector("#end-date");
const responseSchema = "trip-plan.v1";

const today = new Date();
const localToday = new Date(today.getTime() - today.getTimezoneOffset() * 60000)
  .toISOString().slice(0, 10);
startDate.min = localToday;
endDate.min = localToday;

function addDays(dateString, days) {
  const value = new Date(`${dateString}T12:00:00`);
  value.setDate(value.getDate() + days);
  return value.toISOString().slice(0, 10);
}

startDate.addEventListener("change", () => {
  if (!startDate.value) return;
  endDate.min = addDays(startDate.value, 1);
  if (!endDate.value || endDate.value <= startDate.value) {
    endDate.value = addDays(startDate.value, 4);
  }
});

const agentStates = new Map();

function resetLiveView() {
  agentStates.clear();
  eventTimeline.replaceChildren();
  progressPanel.classList.remove("is-complete");
  progressTitle.textContent = "Building your trip";
  progressCopy.textContent = "Connecting the Concierge to the specialist network.";
  document.querySelectorAll(".agent-node").forEach((node) => {
    node.classList.remove("queued", "requested", "submitted", "working", "completed", "failed");
  });
  document.querySelectorAll(".graph-edge").forEach((edge) => {
    edge.classList.remove("active", "completed", "failed");
  });
  document.querySelectorAll("[data-status-for]").forEach((label) => {
    label.textContent = label.dataset.statusFor === "concierge" ? "Queued" : "Waiting";
  });
}

function agentForTool(toolName) {
  if (toolName === "call_travel_agent") return "travel";
  if (toolName === "call_local_experience_agent") return "local";
  if (toolName === "call_transit_agent") return "transit";
  return null;
}

function setAgentState(agent, status) {
  const node = document.querySelector(`.agent-node[data-agent="${agent}"]`);
  if (!node) return;
  const normalized = status === "requested" ? "requested" : status;
  node.classList.remove("queued", "requested", "submitted", "working", "completed", "failed");
  node.classList.add(normalized);
  agentStates.set(agent, normalized);
  const label = document.querySelector(`[data-status-for="${agent}"]`);
  if (label) label.textContent = normalized.replaceAll("_", " ");

  if (agent !== "concierge") {
    const edge = document.querySelector(`.graph-edge[data-edge="${agent}"]`);
    if (edge) {
      edge.classList.remove("active", "completed", "failed");
      edge.classList.add(normalized === "completed" ? "completed" : normalized === "failed" ? "failed" : "active");
    }
  }
}

function appendTimelineEvent(event) {
  const item = document.createElement("li");
  item.className = `timeline-event event-${event.status || "info"}`;
  const dot = document.createElement("span");
  dot.className = "timeline-dot";
  const body = document.createElement("span");
  body.className = "timeline-body";
  const heading = document.createElement("span");
  heading.className = "timeline-message";
  heading.textContent = event.message;
  const meta = document.createElement("span");
  meta.className = "timeline-meta";
  const time = new Date(event.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  meta.textContent = `${event.agent} · ${time}`;
  body.append(heading, meta);

  const preview = event.data?.preview;
  if (preview) {
    const detail = document.createElement("span");
    detail.className = "timeline-preview";
    detail.textContent = preview;
    body.append(detail);
  }
  item.append(dot, body);
  eventTimeline.prepend(item);
  while (eventTimeline.children.length > 24) eventTimeline.lastElementChild.remove();
}

function applyPlannerEvent(event) {
  let agent = event.agent;
  if (event.kind === "tool.called") {
    const delegatedAgent = agentForTool(event.data?.tool_name);
    if (delegatedAgent) {
      agent = delegatedAgent;
      setAgentState(agent, "requested");
    }
  } else if (event.status && agentStates.get(agent) !== "completed") {
    setAgentState(agent, event.status);
  }

  if (event.kind === "agent.artifact") {
    progressTitle.textContent = `${agent[0].toUpperCase()}${agent.slice(1)} research landed`;
    progressCopy.textContent = "The Concierge can now fold this specialist response into the itinerary.";
  } else if (event.kind === "tool.called") {
    progressTitle.textContent = "Concierge is delegating";
    progressCopy.textContent = event.message;
  } else if (event.kind === "agent.status" && event.status === "working") {
    progressTitle.textContent = `${agent[0].toUpperCase()}${agent.slice(1)} is working`;
    progressCopy.textContent = "Live status received from the agent network.";
  }
  appendTimelineEvent(event);
}

function streamPlan(streamUrl) {
  return new Promise((resolve, reject) => {
    const source = new EventSource(streamUrl);
    let settled = false;

    source.addEventListener("planner", (message) => {
      const event = JSON.parse(message.data);
      applyPlannerEvent(event);
      if (event.kind === "plan.completed") {
        settled = true;
        source.close();
        resolve(event.data.result);
      } else if (event.kind === "plan.failed") {
        settled = true;
        source.close();
        reject(new Error(event.message));
      }
    });
    source.onerror = () => {
      if (settled) return;
      settled = true;
      source.close();
      reject(new Error("The live planning stream disconnected. Please try again."));
    };
  });
}

function selectedValues(name) {
  return [...form.querySelectorAll(`input[name="${name}"]:checked`)].map((input) => input.value);
}

function validateChoices() {
  const interests = selectedValues("interests");
  const modes = selectedValues("transport_modes");
  interestError.textContent = interests.length ? "" : "Select at least one interest.";
  transportError.textContent = modes.length ? "" : "Select at least one transport mode.";
  return interests.length > 0 && modes.length > 0;
}

function formatDate(dateString) {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${dateString}T00:00:00Z`));
}

function renderTripSummary(plan) {
  const items = [
    ["Route", `${plan.origin} → ${plan.destination}`],
    ["Dates", `${formatDate(plan.start_date)} – ${formatDate(plan.end_date)}`],
    ["Travelers", String(plan.travelers)],
    ["Budget", `$${Number(plan.budget.total_budget_usd).toLocaleString("en-US")}`],
  ];
  resultSummary.replaceChildren();
  for (const [label, value] of items) {
    const item = document.createElement("div");
    const itemLabel = document.createElement("span");
    const itemValue = document.createElement("strong");
    itemLabel.textContent = label;
    itemValue.textContent = value;
    item.append(itemLabel, itemValue);
    resultSummary.append(item);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formMessage.textContent = "";
  if (!form.reportValidity() || !validateChoices()) return;

  const data = new FormData(form);
  const payload = {
    origin: data.get("origin"),
    destination: data.get("destination"),
    start_date: data.get("start_date"),
    end_date: data.get("end_date"),
    travelers: Number(data.get("travelers")),
    budget_usd: Number(data.get("budget_usd")),
    interests: selectedValues("interests"),
    lodging_style: data.get("lodging_style"),
    pace: data.get("pace"),
    transport_modes: selectedValues("transport_modes"),
    notes: data.get("notes") || null,
  };

  submitButton.disabled = true;
  planner.hidden = true;
  resultPanel.hidden = true;
  progressPanel.hidden = false;
  progressPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  resetLiveView();

  try {
    const response = await fetch("/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Prefer": "respond-async" },
      body: JSON.stringify(payload),
    });
    const body = await response.json();
    if (!response.ok) throw new Error(body.error || "The trip could not be planned.");
    const result = response.status === 202 ? await streamPlan(body.stream_url) : body;
    if (result.schema_version !== responseSchema || !result.plan) {
      throw new Error(
        "The web server is running an outdated response format. Restart it and refresh this page."
      );
    }

    progressPanel.classList.add("is-complete");
    progressTitle.textContent = "Itinerary assembled";
    progressCopy.textContent = "Specialist research has been consolidated into one trip plan.";
    await new Promise((resolve) => window.setTimeout(resolve, 700));
    progressPanel.hidden = true;
    resultTitle.textContent = result.plan.title;
    window.TripPlanRenderer.render(result.plan, resultContent);
    renderTripSummary(result.plan);
    resultMeta.textContent = `Coordinated across the agent network in ${result.duration_seconds.toFixed(1)} seconds.`;
    resultPanel.hidden = false;
    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    progressPanel.hidden = true;
    planner.hidden = false;
    formMessage.textContent = error.message;
    planner.scrollIntoView({ behavior: "smooth", block: "start" });
  } finally {
    submitButton.disabled = false;
  }
});

document.querySelector("#start-over").addEventListener("click", () => {
  resultPanel.hidden = true;
  planner.hidden = false;
  planner.scrollIntoView({ behavior: "smooth", block: "start" });
});

(function attachTripPlanRenderer(root) {
  "use strict";

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  function money(value) {
    if (value === null || value === undefined) return "Not available";
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 0,
    }).format(value);
  }

  function dateLabel(value, options = {}) {
    return new Intl.DateTimeFormat("en-US", {
      month: "short",
      day: "numeric",
      timeZone: "UTC",
      ...options,
    }).format(new Date(`${value}T00:00:00Z`));
  }

  function section(title, className = "") {
    const node = element("section", `itinerary-section ${className}`.trim());
    node.append(element("h2", "", title));
    return node;
  }

  function paragraph(text, className = "") {
    return element("p", className, text);
  }

  function renderAlerts(alerts) {
    if (!alerts.length) return null;
    const wrapper = element("div", "plan-alerts");
    for (const alert of alerts) {
      const card = element("div", `plan-alert alert-${alert.severity}`);
      const content = element("div", "plan-alert-content");
      content.append(element("strong", "", alert.title));
      content.append(paragraph(alert.message));
      if (alert.source) content.append(element("span", "alert-source", alert.source));
      card.append(element("span", "alert-mark", alert.severity === "error" ? "!" : "i"));
      card.append(content);
      wrapper.append(card);
    }
    return wrapper;
  }

  function metric(label, value, className = "") {
    const card = element("div", `plan-metric ${className}`.trim());
    card.append(element("span", "", label));
    card.append(element("strong", "", value));
    return card;
  }

  function renderBudget(budget) {
    const node = section("Budget snapshot", "budget-section");
    const metrics = element("div", "budget-metrics");
    metrics.append(metric("Total budget", money(budget.total_budget_usd)));
    metrics.append(metric("Flights", money(budget.flight_estimate_usd)));
    metrics.append(metric("Lodging", money(budget.lodging_estimate_usd)));
    metrics.append(
      metric(
        "Remaining",
        money(budget.remaining_usd),
        `budget-${budget.status}`,
      ),
    );
    node.append(metrics);
    if (budget.notes.length) node.append(renderTextList(budget.notes));
    return node;
  }

  function optionMeta(items) {
    const meta = element("div", "option-meta");
    for (const item of items.filter(Boolean)) meta.append(element("span", "", item));
    return meta;
  }

  function renderFlights(flights) {
    const node = section("Flights", "travel-section");
    node.append(paragraph(flights.summary, "section-summary"));
    if (!flights.options.length) return node;

    const grid = element("div", "option-grid");
    flights.options.forEach((flight, index) => {
      const recommended = index === flights.recommended_option_index;
      const card = element("article", `option-card${recommended ? " recommended" : ""}`);
      if (recommended) card.append(element("span", "recommendation-badge", "Recommended"));
      card.append(element("h3", "", [flight.airline, flight.flight_number].filter(Boolean).join(" ")));
      card.append(optionMeta([
        `${flight.departure} → ${flight.arrival}`,
        flight.stops === null ? null : `${flight.stops} stop${flight.stops === 1 ? "" : "s"}`,
      ]));
      card.append(element("strong", "option-price", money(flight.price_usd)));
      if (flight.note) card.append(paragraph(flight.note));
      grid.append(card);
    });
    node.append(grid);
    return node;
  }

  function renderHotels(hotels) {
    const node = section("Where to stay", "travel-section");
    node.append(paragraph(hotels.summary, "section-summary"));
    if (!hotels.options.length) return node;

    const grid = element("div", "option-grid");
    hotels.options.forEach((hotel, index) => {
      const recommended = index === hotels.recommended_option_index;
      const card = element("article", `option-card${recommended ? " recommended" : ""}`);
      if (recommended) card.append(element("span", "recommendation-badge", "Recommended"));
      card.append(element("h3", "", hotel.name));
      card.append(optionMeta([
        hotel.neighborhood,
        hotel.rating === null ? null : `${hotel.rating.toFixed(1)} rating`,
        hotel.price_per_night_usd === null ? null : `${money(hotel.price_per_night_usd)}/night`,
      ]));
      card.append(element("strong", "option-price", money(hotel.total_price_usd)));
      card.append(paragraph(hotel.why_it_fits));
      if (hotel.caveat) card.append(paragraph(hotel.caveat, "option-caveat"));
      grid.append(card);
    });
    node.append(grid);
    return node;
  }

  function rainLabel(chance) {
    if (chance === null || chance === undefined) return null;
    return `${Math.round(chance * 100)}% rain`;
  }

  function renderWeather(weather) {
    const node = section("Weather", "weather-section");
    node.append(paragraph(weather.summary, "section-summary"));

    if (weather.days.length) {
      const strip = element("div", "weather-strip");
      for (const day of weather.days) {
        const card = element("article", "weather-card");
        card.append(element("span", "weather-date", dateLabel(day.day, { weekday: "short" })));
        const temperatures = day.high_celsius === null || day.low_celsius === null
          ? "Forecast unavailable"
          : `${Math.round(day.high_celsius)}° / ${Math.round(day.low_celsius)}°C`;
        card.append(element("strong", "", temperatures));
        card.append(paragraph(day.description));
        const rain = rainLabel(day.chance_of_rain);
        if (rain) card.append(element("span", "weather-rain", rain));
        strip.append(card);
      }
      node.append(strip);
    }

    if (weather.packing_advice.length) {
      const advice = element("div", "packing-advice");
      advice.append(element("strong", "", "Pack for the trip"));
      advice.append(renderTextList(weather.packing_advice));
      node.append(advice);
    }
    return node;
  }

  function renderTransportOptions(options) {
    if (!options.length) return null;
    const wrapper = element("div", "transport-chips");
    for (const option of options) {
      const details = [option.mode];
      if (option.duration_minutes !== null) details.push(`${option.duration_minutes} min`);
      if (option.distance_km !== null) details.push(`${option.distance_km.toFixed(1)} km`);
      const chip = element("span", "transport-chip", details.join(" · "));
      if (option.note) chip.title = option.note;
      wrapper.append(chip);
    }
    return wrapper;
  }

  function renderDays(days) {
    const wrapper = element("div", "days-list");
    for (const day of days) {
      const node = section(
        `${dateLabel(day.day, { weekday: "long" })} · ${day.title}`,
        "day-section",
      );
      if (day.weather_note) node.append(paragraph(day.weather_note, "day-weather-note"));

      const timeline = element("div", "activity-timeline");
      for (const activity of day.activities) {
        const item = element("article", "activity-card");
        item.append(element("span", "activity-period", activity.period));
        item.append(element("h3", "", activity.title));
        item.append(paragraph(activity.description));
        const meta = optionMeta([
          activity.location,
          activity.estimated_cost_usd === null ? null : money(activity.estimated_cost_usd),
        ]);
        if (meta.childNodes.length) item.append(meta);
        const transport = renderTransportOptions(activity.transport_options);
        if (transport) item.append(transport);
        timeline.append(item);
      }
      node.append(timeline);
      wrapper.append(node);
    }
    return wrapper;
  }

  function renderTextList(items) {
    const list = element("ul", "structured-list");
    for (const item of items) list.append(element("li", "", item));
    return list;
  }

  function renderSimpleSection(title, items, className) {
    if (!items.length) return null;
    const node = section(title, className);
    node.append(renderTextList(items));
    return node;
  }

  function render(plan, target) {
    const fragment = document.createDocumentFragment();
    fragment.append(paragraph(plan.summary, "plan-summary"));

    const alerts = renderAlerts(plan.alerts);
    if (alerts) fragment.append(alerts);
    fragment.append(renderBudget(plan.budget));

    const travelGrid = element("div", "travel-grid");
    travelGrid.append(renderFlights(plan.flights));
    travelGrid.append(renderHotels(plan.hotels));
    fragment.append(travelGrid);

    fragment.append(renderWeather(plan.weather));
    fragment.append(renderDays(plan.days));

    const strategy = renderSimpleSection(
      "Getting around",
      plan.transport_strategy,
      "strategy-section",
    );
    if (strategy) fragment.append(strategy);

    const nextSteps = renderSimpleSection(
      "What to book next",
      plan.next_steps,
      "next-steps-section",
    );
    if (nextSteps) fragment.append(nextSteps);

    target.replaceChildren(fragment);
  }

  root.TripPlanRenderer = { render };
})(window);

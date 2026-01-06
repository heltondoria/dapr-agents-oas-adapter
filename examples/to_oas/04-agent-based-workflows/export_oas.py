"""Exporta specs OAS (YAML) dos agentes deste exemplo."""

from __future__ import annotations

from pathlib import Path

from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import DaprAgentConfig


def _build_agents() -> list[DaprAgentConfig]:
    return [
        DaprAgentConfig(
            name="DestinationExtractor",
            role="Extract destination",
            goal="Extract the destination city from a user message.",
            instructions=[
                "Extract the main city from the user's message.",
                "Return only the city name, nothing else.",
            ],
            tools=[],
        ),
        DaprAgentConfig(
            name="PlannerAgent",
            role="Trip planner",
            goal="Create a concise 3-day trip outline for the given destination.",
            instructions=[
                "Create a concise 3-day outline for the given destination.",
                "Balance culture, food, and leisure activities.",
            ],
            tools=[],
        ),
        DaprAgentConfig(
            name="ItineraryAgent",
            role="Itinerary expander",
            goal="Expand a trip outline into a detailed itinerary.",
            instructions=[
                "Expand a 3-day outline into a detailed itinerary.",
                "Include Morning, Afternoon, and Evening sections each day.",
            ],
            tools=[],
        ),
    ]


def export_all(*, out_dir: Path) -> list[Path]:
    exporter = DaprAgentSpecExporter()
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    for agent in _build_agents():
        target = out_dir / f"{agent.name}.yaml"
        exporter.to_yaml_file(agent, target)
        outputs.append(target)

    return outputs


def main() -> int:
    out_dir = Path(__file__).parent / "exported"
    for path in export_all(out_dir=out_dir):
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

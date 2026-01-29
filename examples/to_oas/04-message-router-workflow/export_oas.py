"""Export the message-router workflow spec as OAS (YAML)."""

from __future__ import annotations

from pathlib import Path

from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import (
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)


def _build_blog_workflow_definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="blog_workflow",
        description="Workflow triggered by Pub/Sub that creates an outline and writes a post.",
        inputs=[{"title": "topic", "type": "string"}],
        outputs=[{"title": "post", "type": "string"}],
        tasks=[
            WorkflowTaskDefinition(
                name="start", task_type="start", inputs=["topic"], outputs=["topic"]
            ),
            WorkflowTaskDefinition(
                name="create_outline",
                task_type="llm",
                config={
                    "prompt_template": (
                        "Create a short outline about {{ topic }}. Output 3-5 bullet points."
                    )
                },
                inputs=["topic"],
                outputs=["outline"],
            ),
            WorkflowTaskDefinition(
                name="write_post",
                task_type="llm",
                config={
                    "prompt_template": (
                        "Write a short blog post following this outline:\n{{ outline }}"
                    )
                },
                inputs=["outline"],
                outputs=["post"],
            ),
            WorkflowTaskDefinition(name="end", task_type="end", inputs=["post"], outputs=["post"]),
        ],
        edges=[
            WorkflowEdgeDefinition(
                from_node="start", to_node="create_outline", data_mapping={"topic": "topic"}
            ),
            WorkflowEdgeDefinition(
                from_node="create_outline",
                to_node="write_post",
                data_mapping={"outline": "outline"},
            ),
            WorkflowEdgeDefinition(
                from_node="write_post", to_node="end", data_mapping={"post": "post"}
            ),
        ],
        start_node="start",
        end_nodes=["end"],
    )


def export_all(*, out_dir: Path) -> list[Path]:
    exporter = DaprAgentSpecExporter()
    out_dir.mkdir(parents=True, exist_ok=True)

    workflow_def = _build_blog_workflow_definition()
    target = out_dir / "blog_workflow.yaml"
    exporter.to_yaml_file(workflow_def, target)
    return [target]


def main() -> int:
    out_dir = Path(__file__).parent / "exported"
    for path in export_all(out_dir=out_dir):
        print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

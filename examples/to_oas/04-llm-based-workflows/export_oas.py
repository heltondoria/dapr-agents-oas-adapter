"""Export OAS specs (YAML) for this example."""

from __future__ import annotations

from pathlib import Path

from dapr_agents_oas_adapter import DaprAgentSpecExporter
from dapr_agents_oas_adapter.types import (
    WorkflowDefinition,
    WorkflowEdgeDefinition,
    WorkflowTaskDefinition,
)


def _build_single_task_workflow_definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        name="single_task_workflow",
        description="Simple workflow: a single LLM activity.",
        inputs=[{"title": "name", "type": "string"}],
        outputs=[{"title": "bio", "type": "string"}],
        tasks=[
            WorkflowTaskDefinition(
                name="start",
                task_type="start",
                inputs=["name"],
                outputs=["name"],
            ),
            WorkflowTaskDefinition(
                name="describe_person",
                task_type="llm",
                # OAS templates use `{{ var }}` placeholders.
                config={"prompt_template": "Who was {{ name }}?"},
                inputs=["name"],
                outputs=["bio"],
            ),
            WorkflowTaskDefinition(
                name="end",
                task_type="end",
                inputs=["bio"],
                outputs=["bio"],
            ),
        ],
        edges=[
            WorkflowEdgeDefinition(
                from_node="start",
                to_node="describe_person",
                data_mapping={"name": "name"},
            ),
            WorkflowEdgeDefinition(
                from_node="describe_person",
                to_node="end",
                data_mapping={"bio": "bio"},
            ),
        ],
        start_node="start",
        end_nodes=["end"],
    )


def export_all(*, out_dir: Path) -> list[Path]:
    exporter = DaprAgentSpecExporter()
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []

    workflow_def = _build_single_task_workflow_definition()
    target = out_dir / "single_task_workflow.yaml"
    exporter.to_yaml_file(workflow_def, target)
    outputs.append(target)

    return outputs


def main() -> int:
    out_dir = Path(__file__).parent / "exported"
    written = export_all(out_dir=out_dir)
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

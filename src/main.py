"""CLI entry point for the IELTS Task 2 learning agent."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .agent.state import Scenario, WorkflowState
from .agent.workflow import AgentWorkflow
from .llm.factory import LLMFactory
from .prompts.templates import PromptManager
from .rag.embedder import Embedder
from .rag.retriever import Retriever
from .rag.store import VectorStore
from .utils.config import Config
from .utils.logger import setup_logger


console = Console()
logger = setup_logger("ielts_cli")


def build_workflow(provider: str | None = None, model: str | None = None) -> AgentWorkflow:
    config = Config()
    provider_name = provider or config.default_provider
    overrides = {"model_name": model} if model else None
    llm = LLMFactory.create(provider_name, config, overrides)

    retriever = None
    try:
        embedder = Embedder(config.get("rag.embedding_model", "all-MiniLM-L6-v2"))
        store = VectorStore(config.chroma_persist_dir)
        retriever = Retriever(embedder, store)
    except Exception as exc:
        logger.warning("RAG initialization skipped: %s", exc)

    return AgentWorkflow(
        llm=llm,
        retriever=retriever,
        prompt_manager=PromptManager(),
    )


def print_analysis(state: WorkflowState):
    if not state.analysis:
        return
    table = Table(title="Question Breakdown")
    table.add_column("Item", style="cyan")
    table.add_column("Value")
    table.add_row("Type", state.analysis.question_type_en)
    table.add_row("Code", state.analysis.question_type)
    table.add_row("Topics", ", ".join(state.analysis.topics) or "auto")
    console.print(table)
    if state.analysis.controversy:
        console.print(Panel(Markdown(state.analysis.controversy), title="Core Issue"))


def print_errors(state: WorkflowState) -> bool:
    if not state.has_errors():
        return False
    for error in state.errors:
        console.print(f"[red]{error}[/red]")
    return True


@click.group()
@click.option("--provider", "-p", default=None, help="LLM provider: openai or deepseek.")
@click.option("--model", "-m", default=None, help="Override model name.")
@click.pass_context
def cli(ctx, provider, model):
    """IELTS Task 2 idea, outline, and quality-control CLI."""
    ctx.ensure_object(dict)
    ctx.obj["provider"] = provider
    ctx.obj["model"] = model


@cli.command()
@click.argument("question")
@click.option("--stance", "-s", default=None, help="Optional stance for the outline.")
@click.pass_context
def generate(ctx, question, stance):
    """Generate question analysis, ideas, and a suggested outline."""
    workflow = build_workflow(ctx.obj.get("provider"), ctx.obj.get("model"))
    state = WorkflowState(
        user_input=question,
        scenario=Scenario.GENERATE,
        selected_stance=stance,
    )
    state = workflow.run(state)
    print_analysis(state)
    if print_errors(state):
        return
    if state.arguments:
        console.print(Panel(Markdown(state.arguments[0].main_idea_en), title="Writable Ideas"))
    if state.outline and state.outline.tips:
        console.print(Panel(Markdown(state.outline.tips[0]), title="Suggested Outline"))


@cli.command()
@click.argument("idea")
@click.option("--question", "-q", default="", help="Original IELTS question for context.")
@click.pass_context
def deepen(ctx, idea, question):
    """Develop a shallow idea into an IELTS-style argument chain."""
    workflow = build_workflow(ctx.obj.get("provider"), ctx.obj.get("model"))
    state = WorkflowState(
        user_input=question or idea,
        scenario=Scenario.DEEPEN,
        user_idea=idea,
    )
    state = workflow.run(state)
    if print_errors(state):
        return
    if state.arguments:
        console.print(Panel(Markdown(state.arguments[0].main_idea_en), title="Developed Argument"))


@cli.command()
@click.argument("question")
@click.option("--outline", "-o", default=None, help="Path to outline file, or outline text.")
@click.pass_context
def evaluate(ctx, question, outline):
    """Evaluate a student's outline as an idea-and-structure plan."""
    outline_text = outline or ""
    if outline and Path(outline).exists():
        outline_text = Path(outline).read_text(encoding="utf-8")
    if not outline_text:
        console.print("[yellow]Paste the outline, then press Ctrl+Z/Ctrl+D when finished.[/yellow]")
        lines = []
        try:
            while True:
                lines.append(input())
        except EOFError:
            pass
        outline_text = "\n".join(lines)

    workflow = build_workflow(ctx.obj.get("provider"), ctx.obj.get("model"))
    state = WorkflowState(
        user_input=question,
        scenario=Scenario.EVALUATE,
        user_outline=outline_text,
    )
    state = workflow.run(state)
    print_analysis(state)
    if print_errors(state):
        return
    if state.evaluation:
        table = Table(title="Outline Evaluation")
        table.add_column("Criterion")
        table.add_column("Score")
        table.add_row("Task Response", f"{state.evaluation.task_response:.1f}")
        table.add_row("Coherence and Cohesion", f"{state.evaluation.coherence_cohesion:.1f}")
        table.add_row("Lexical Resource", f"{state.evaluation.lexical_resource:.1f}")
        table.add_row("Grammar Range and Accuracy", f"{state.evaluation.grammatical_range:.1f}")
        table.add_row("Overall", f"{state.evaluation.overall_band:.1f}")
        console.print(table)
        for suggestion in state.evaluation.suggestions:
            console.print(Markdown(suggestion))


@cli.command()
@click.option("--topic", "-t", default=None)
@click.pass_context
def practice(ctx, topic):
    """Pick a question from the topic bank and run idea generation."""
    questions_file = Path(__file__).parent.parent / "data" / "test_cases" / "ielts_questions.json"
    if not questions_file.exists():
        console.print("[red]No question bank found.[/red]")
        return

    data = json.loads(questions_file.read_text(encoding="utf-8-sig"))
    questions = data.get("questions", [])
    if topic:
        questions = [
            item for item in questions
            if topic.lower() in [t.lower() for t in item.get("topics", [])]
        ]
    if not questions:
        console.print("[red]No matching practice question found.[/red]")
        return

    question = random.choice(questions)["question"]
    console.print(Panel(question, title="Practice Question"))
    ctx.invoke(generate, question=question, stance=None)


def main():
    if len(sys.argv) == 1:
        cli(["--help"])
        return
    cli()


if __name__ == "__main__":
    main()

"""CLI 入口 — ml-opt 命令行工具"""

import os
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(name="ml-opt", help="Multi-agent ML experiment optimization system")
console = Console()


@app.command()
def optimize(
    repo_path: str = typer.Argument(..., help="Path to the ML training repo"),
    goal: str = typer.Option("maximize val_accuracy", "--goal", "-g", help="Optimization goal"),
    max_trials: int = typer.Option(3, "--max-trials", "-n", help="Maximum number of trials"),
    max_hours: float = typer.Option(6.0, "--max-hours", help="Maximum time budget in hours"),
    metric: str = typer.Option("val_accuracy", "--metric", "-m", help="Primary metric to optimize"),
    threshold: float = typer.Option(0.01, "--threshold", help="Improvement threshold"),
    use_llm: bool = typer.Option(False, "--llm", help="Use LLM agents (requires OPENAI_API_KEY)"),
    llm_model: str = typer.Option("gpt-4o-mini", "--model", help="LLM model to use"),
    output_dir: str = typer.Option("ml_opt_output", "--output", "-o", help="Output directory for reports"),
):
    """Run ML experiment optimization on a training repo."""
    repo_path = str(Path(repo_path).resolve())
    if not Path(repo_path).is_dir():
        console.print(f"[red]Error: repo path not found: {repo_path}[/red]")
        raise typer.Exit(1)

    console.print(Panel.fit(
        f"[bold]ML Experiment Optimizer[/bold]\n"
        f"Repo: {repo_path}\n"
        f"Goal: {goal}\n"
        f"Metric: {metric} (threshold={threshold})\n"
        f"Budget: {max_trials} trials, {max_hours}h\n"
        f"LLM: {'on' if use_llm else 'off'}",
        title="Configuration",
    ))

    from schemas.task import TaskSpec, BudgetSpec, SuccessCriteria
    from schemas.common import RiskLevel
    from tools import build_default_registry
    from tools.guard import ToolGuard
    from workflow.graph import create_runnable
    from observability import TraceLogger, ReportGenerator

    task = TaskSpec(
        task_id=f"task_{os.urandom(4).hex()}",
        repo_path=repo_path,
        goal=goal,
        budget=BudgetSpec(max_trials=max_trials, max_hours=max_hours),
        success_criteria=SuccessCriteria(
            primary_metric=metric,
            improvement_threshold=threshold,
        ),
    )

    registry = build_default_registry()
    guard = ToolGuard()
    guard.set_auto_approve(RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH)

    repo_agent = None
    planner_agent = None
    if use_llm:
        from llm import LLMAdapter, TokenBudget
        from agents import RepoAgent, PlannerAgent
        from tools.dispatcher import ToolDispatcher

        llm = LLMAdapter(model=llm_model)
        budget = TokenBudget()
        dispatcher_for_agents = ToolDispatcher(registry, guard)
        repo_agent = RepoAgent(llm=llm, dispatcher=dispatcher_for_agents, token_budget=budget)
        planner_agent = PlannerAgent(llm=llm, token_budget=budget)

    runnable, dispatcher = create_runnable(
        registry, guard,
        repo_agent=repo_agent,
        planner_agent=planner_agent,
        use_llm=use_llm,
    )

    trace = TraceLogger(task.task_id, output_dir=output_dir)

    console.print("\n[bold green]Starting optimization...[/bold green]\n")
    result = runnable.invoke({"task_spec": task})

    run_state = result["run_state"]
    messages = result.get("messages", [])

    for msg in messages:
        role = msg.get("role", "system")
        content = msg.get("content", "")
        color = {
            "system": "dim",
            "supervisor": "bold yellow",
            "repo_agent": "cyan",
            "planner_agent": "green",
            "executor_agent": "blue",
            "critic_agent": "magenta",
        }.get(role, "white")
        console.print(f"  [{color}][{role}][/{color}] {content}")

    trace.log_from_messages(messages)
    trace.log_tool_calls(dispatcher.call_records)
    trace_path = trace.save()

    report_gen = ReportGenerator()
    report = report_gen.generate(task, run_state, messages, dispatcher.call_records, trace_path)
    report_path = Path(output_dir) / f"report_{task.task_id}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    console.print(f"\n{'='*60}")
    table = Table(title="Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Status", run_state.phase.value)
    table.add_row("Trials", str(run_state.trials_completed))
    if run_state.current_best:
        table.add_row(f"Best {run_state.current_best.metric_name}", str(run_state.current_best.metric_value))
    table.add_row("Tool calls", str(len(dispatcher.call_records)))
    success_rate = sum(1 for r in dispatcher.call_records if r.output_status.value == "success") / max(len(dispatcher.call_records), 1)
    table.add_row("Tool success rate", f"{success_rate:.0%}")
    if run_state.total_tokens_used:
        table.add_row("LLM tokens", f"{run_state.total_tokens_used:,}")
    console.print(table)
    console.print(f"\n[dim]Report: {report_path}[/dim]")
    console.print(f"[dim]Trace:  {trace_path}[/dim]")


@app.command()
def status(
    output_dir: str = typer.Option("ml_opt_output", "--output", "-o"),
):
    """Show recent optimization results."""
    output = Path(output_dir)
    if not output.exists():
        console.print("[dim]No results found.[/dim]")
        return

    reports = sorted(output.glob("report_*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not reports:
        console.print("[dim]No reports found.[/dim]")
        return

    for r in reports[:5]:
        console.print(f"  {r.name} ({r.stat().st_size} bytes)")


@app.command()
def history(
    output_dir: str = typer.Option("ml_opt_output", "--output", "-o"),
):
    """Show trace history."""
    output = Path(output_dir)
    traces = sorted(output.glob("trace_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not traces:
        console.print("[dim]No traces found.[/dim]")
        return

    import json
    for t in traces[:5]:
        data = json.loads(t.read_text())
        console.print(
            f"  {data['task_id']} — {data['total_events']} events, "
            f"{data['total_duration_sec']}s"
        )


if __name__ == "__main__":
    app()

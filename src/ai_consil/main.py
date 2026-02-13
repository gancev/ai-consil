"""Main entry point for ai-consil: FastAPI app and Typer CLI."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_consil import __version__
from ai_consil.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# FastAPI application
app = FastAPI(
    title="AI Council",
    description="Multi-agent deliberation with blind voting via OpenAI-compatible API",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


# Typer CLI application
cli_app = typer.Typer(
    name="ai-consil",
    help="AI Council - Multi-agent deliberation with blind voting",
    add_completion=False,
)


@cli_app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    config: str = typer.Option(
        None, "--config", "-c", help="Default council config JSON file"
    ),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
    log_level: str = typer.Option("info", "--log-level", "-l", help="Log level"),
) -> None:
    """Start the AI Council API server."""
    if config:
        import os
        os.environ["AI_CONSIL_DEFAULT_CONFIG"] = config
        typer.echo(f"Default config: {config}")

    typer.echo(f"Starting AI Council server v{__version__}")
    typer.echo(f"Listening on http://{host}:{port}")
    typer.echo(f"API docs: http://{host}:{port}/docs")

    uvicorn.run(
        "ai_consil.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


@cli_app.command()
def run(
    topic: str = typer.Argument(..., help="Topic or question for the council"),
    config: str = typer.Option(
        None, "--config", "-c", help="Path to council config JSON file"
    ),
    output: str = typer.Option(
        "./out", "--output", "-o", help="Output directory for artifacts"
    ),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream output"),
    transcript: bool = typer.Option(
        False, "--transcript", "-t", help="Generate a human-readable markdown transcript report"
    ),
) -> None:
    """Run a council deliberation from the command line."""
    from ai_consil.config.loader import get_default_config, load_config_from_file
    from ai_consil.council.orchestrator import CouncilOrchestrator
    from ai_consil.storage.artifacts import ArtifactStore

    # Load config
    if config:
        try:
            council_config = load_config_from_file(config)
        except Exception as e:
            typer.echo(f"Error loading config: {e}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo("No config provided, using default mock configuration")
        council_config = get_default_config()

    # Create orchestrator
    orchestrator = CouncilOrchestrator(config=council_config)
    store = ArtifactStore(output)

    typer.echo(f"\nSession ID: {orchestrator.session_id}")
    typer.echo(f"Topic: {topic}")
    typer.echo(f"Agents: {[a.id for a in council_config.agents]}")
    typer.echo(f"Rounds: {council_config.rounds}")
    typer.echo("-" * 50)

    # Run deliberation
    async def _run():
        if stream:
            collected_events: list = []
            async for event in orchestrator.run_stream(topic):
                store.append_event(orchestrator.session_id, event)
                collected_events.append(event)
                _print_event(event)
            final = orchestrator.state.final_answer if orchestrator.state else ""
            if transcript:
                report_path = store.write_transcript_report(
                    orchestrator.session_id, collected_events, topic
                )
                typer.echo(f"\nTranscript report: {report_path}")
            return final
        else:
            final_answer, trace, events = await orchestrator.run(topic)
            store.write_vote_ledger(orchestrator.session_id, trace)
            if transcript:
                report_path = store.write_transcript_report(
                    orchestrator.session_id, events, topic
                )
                typer.echo(f"\nTranscript report: {report_path}")
            return final_answer

    final_answer = asyncio.run(_run())

    typer.echo("-" * 50)
    typer.echo("\nFinal Answer:")
    typer.echo(final_answer)
    typer.echo(f"\nArtifacts saved to: {store.get_session_path(orchestrator.session_id)}")


def _print_event(event: Any) -> None:
    """Print a council event to the console."""
    event_type = event.event_type.value
    timestamp = event.timestamp

    if event_type == "session_start":
        typer.echo(f"\n[{timestamp}] Session started")
    elif event_type == "round_start":
        typer.echo(f"\n[{timestamp}] === Round {event.round} ===")
    elif event_type == "agent_analysis":
        typer.echo(f"\n[{timestamp}] {event.agent_id} analysis:")
        typer.echo(f"  {event.content[:200]}..." if event.content else "  (empty)")
    elif event_type == "question":
        typer.echo(f"\n[{timestamp}] Q: {event.from_agent} -> {event.to_agent}")
        typer.echo(f"  {event.content}")
    elif event_type == "answer":
        typer.echo(f"\n[{timestamp}] A: {event.from_agent} -> {event.to_agent}")
        typer.echo(f"  {event.content[:200]}..." if event.content else "  (empty)")
    elif event_type == "voting_open":
        typer.echo(f"\n[{timestamp}] Voting opened for round {event.round}")
    elif event_type == "voting_closed":
        typer.echo(f"\n[{timestamp}] Voting closed for round {event.round}")
    elif event_type == "vote_reveal":
        typer.echo(f"\n[{timestamp}] Vote reveal for round {event.round}:")
        if event.tally and event.tally.counts:
            parts = [f"{option}: {count}" for option, count in event.tally.counts.items()]
            typer.echo(f"  {', '.join(parts)}")
    elif event_type == "synthesis":
        typer.echo(f"\n[{timestamp}] Synthesis:")
        typer.echo(f"  {event.content[:300]}..." if event.content else "  (empty)")
    elif event_type == "session_end":
        typer.echo(f"\n[{timestamp}] Session ended")
    elif event_type == "error":
        typer.echo(f"\n[{timestamp}] ERROR: {event.error}", err=True)


@cli_app.command()
def providers() -> None:
    """List available LLM providers."""
    from ai_consil.providers import registry

    typer.echo("Available Providers:")
    typer.echo("-" * 30)

    for name in registry.list_providers():
        available = "✓" if name in registry.list_available() else "✗"
        typer.echo(f"  {available} {name}")

    typer.echo("\nLegend: ✓ = configured, ✗ = missing env vars")


@cli_app.command()
def version() -> None:
    """Show version information."""
    typer.echo(f"ai-consil v{__version__}")


def main() -> None:
    """Main entry point."""
    cli_app()


if __name__ == "__main__":
    main()

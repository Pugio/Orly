"""Benchmark script: start backend, run scenarios, report latency.

Usage:
    uv run python -m simulation.latency_benchmark --backend ws://localhost:8080/ws/session
    uv run python -m simulation.latency_benchmark --start-backend
    uv run python -m simulation.latency_benchmark --backend ws://localhost:8080/ws/session --scenario simple_question
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import time

from simulation.scenarios import ALL_SCENARIOS, Scenario
from simulation.sim_client import SimClient, format_latency, print_result, ScenarioResult

logger = logging.getLogger(__name__)


def start_backend_server(host: str = "0.0.0.0", port: int = 8080) -> subprocess.Popen:
    """Start the FastAPI backend as a subprocess.

    Returns the Popen handle so the caller can terminate it.
    """
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    logger.info("Starting backend: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the server to be ready.
    for _ in range(30):
        time.sleep(1)
        if proc.poll() is not None:
            stdout = proc.stdout.read().decode() if proc.stdout else ""
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(
                f"Backend failed to start.\nstdout: {stdout}\nstderr: {stderr}"
            )
        # Try connecting.
        try:
            import urllib.request

            urllib.request.urlopen(f"http://{host}:{port}/docs", timeout=2)
            logger.info("Backend is ready.")
            return proc
        except Exception:
            continue

    raise RuntimeError("Backend did not become ready within 30 seconds.")


def print_summary_table(results: list[ScenarioResult]):
    """Print a summary table of all scenario results."""
    print(f"\n{'=' * 80}")
    print("LATENCY BENCHMARK SUMMARY")
    print(f"{'=' * 80}")
    header = f"{'Scenario':<20} {'Round-trip':>12} {'-> Transcript':>14} {'-> Audio':>12} {'-> Tool':>12} {'Events':>8}"
    print(header)
    print("-" * 80)
    for r in results:
        row = (
            f"{r.scenario_name:<20} "
            f"{format_latency(r.total_round_trip_ms):>12} "
            f"{format_latency(r.send_to_transcript_out_ms):>14} "
            f"{format_latency(r.send_to_audio_response_ms):>12} "
            f"{format_latency(r.send_to_tool_result_ms):>12} "
            f"{len(r.events):>8}"
        )
        print(row)
    print(f"{'=' * 80}")


async def run_benchmark(
    backend_url: str,
    scenarios: list[Scenario],
) -> list[ScenarioResult]:
    """Run all scenarios and return results."""
    results = []

    for scenario in scenarios:
        client = SimClient(backend_url)
        try:
            await client.connect()
            result = await client.run_scenario(scenario)
            results.append(result)
            print_result(result)
        except Exception as e:
            logger.exception("Failed to run scenario %s", scenario.name)
            results.append(
                ScenarioResult(scenario_name=scenario.name, error=str(e))
            )
        finally:
            await client.close()

        # Brief pause between scenarios to let the backend reset.
        await asyncio.sleep(2.0)

    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orly latency benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="ws://localhost:8080/ws/session",
        help="Backend WebSocket URL (default: ws://localhost:8080/ws/session)",
    )
    parser.add_argument(
        "--start-backend",
        action="store_true",
        help="Start the backend server automatically",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        choices=list(ALL_SCENARIOS.keys()),
        help="Run a single scenario (default: run all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    backend_proc = None

    if args.start_backend:
        backend_proc = start_backend_server()

    try:
        if args.scenario:
            scenarios = [ALL_SCENARIOS[args.scenario]]
        else:
            scenarios = list(ALL_SCENARIOS.values())

        results = asyncio.run(run_benchmark(args.backend, scenarios))
        print_summary_table(results)

    finally:
        if backend_proc:
            logger.info("Stopping backend...")
            backend_proc.terminate()
            backend_proc.wait(timeout=5)


if __name__ == "__main__":
    main()

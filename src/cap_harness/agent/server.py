"""Standalone System 2 server (YAM or G1), also embeddable as an ASGI application."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import threading
from typing import Any

from .embedded import EmbeddedAgentRuntime
from .mcp import create_lifespan, create_mcp_transport


def create_app(session: Any) -> Any:
    """Serve MCP and operator steering without creating another environment."""
    from fastapi import FastAPI
    from starlette.routing import Mount

    manager, endpoint = create_mcp_transport(session)
    app = FastAPI(lifespan=create_lifespan(manager))
    app.router.routes.append(Mount("/mcp", app=endpoint))

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "job": session.job()}

    @app.post("/steer")
    def steer(payload: dict[str, str]) -> dict[str, Any]:
        event = session.steer(payload["message"])
        return {"ok": True, "seq": event.seq}

    return app


def build_runtime(
    env: Any,
    *,
    adapter_type: type,
    public_tool_names: frozenset[str],
    family: str,
    allow_motion: bool = False,
    episode_recorder: Any = None,
    event_sink: Any = None,
    tool_result_sink: Any = None,
    **api_options: Any,
) -> tuple[Any, EmbeddedAgentRuntime]:
    """Attach to a caller-owned station; construction performs no motion."""
    from cap_harness.api import CapApi
    from cap_harness.contracts import TaskContext
    from cap_harness.registry import ToolRegistry
    from cap_harness.yam_real.control.runtime import RuntimeControl
    from cap_harness.yam_real.live_adapter import LiveYamAdapter

    raw = adapter_type(
        env,
        allow_physical_motion=allow_motion,
        task_context=TaskContext(
            suite="system2",
            task_id=0,
            task_name="system2",
            language="Follow the operator goal.",
            family=family,
        ),
    )
    control = RuntimeControl(raw)
    control.episode_recorder = episode_recorder
    adapter = LiveYamAdapter(raw, control)
    try:
        api = CapApi(
            adapter,
            default_camera=env.config.camera.role,
            **api_options,
        )
        registry = api.register_tools(ToolRegistry(public_extension_allowlist=public_tool_names))
        runtime = EmbeddedAgentRuntime(
            api,
            registry,
            operation_controller=control,
            event_sink=event_sink,
            tool_result_sink=tool_result_sink,
        )
    except BaseException:
        control.close()
        raise
    return adapter, runtime


def build_yam_runtime(env: Any, **options: Any) -> tuple[Any, EmbeddedAgentRuntime]:
    """``build_runtime`` with the YAM adapter and public tool allowlist."""
    from cap_harness.registry import YAM_REAL_PUBLIC_TOOL_NAMES
    from cap_harness.yam_real.adapter import YamRealAdapter

    return build_runtime(
        env,
        adapter_type=YamRealAdapter,
        public_tool_names=YAM_REAL_PUBLIC_TOOL_NAMES,
        family="yam_real",
        **options,
    )


def _real_embodiment(benchmark: str) -> tuple[Any, ...]:
    """(adapter_type, public_tool_names, real_station, offline_station, kinematics_type)."""
    if benchmark == "yam_real":
        from cap_harness.registry import YAM_REAL_PUBLIC_TOOL_NAMES
        from cap_harness.yam_real.adapter import YamRealAdapter
        from cap_harness.yam_real.kinematics import YamKinematics
        from cap_harness.yam_real.station import build_real_station, build_sim_station

        return (
            YamRealAdapter,
            YAM_REAL_PUBLIC_TOOL_NAMES,
            build_real_station,
            build_sim_station,
            YamKinematics,
        )
    from cap_harness.g1_real.adapter import G1RealAdapter
    from cap_harness.g1_real.kinematics import G1Kinematics
    from cap_harness.g1_real.station import build_fake_station, build_real_station
    from cap_harness.registry import G1_REAL_PUBLIC_TOOL_NAMES

    return (
        G1RealAdapter,
        G1_REAL_PUBLIC_TOOL_NAMES,
        build_real_station,
        build_fake_station,
        G1Kinematics,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("yam_real", "g1_real"), default="yam_real")
    parser.add_argument("--station", default="yam-example")
    parser.add_argument("--station-config-root", type=Path)
    parser.add_argument("--sim", action="store_true", help="Use the off-robot YAM station")
    parser.add_argument("--fake", action="store_true", help="Use the off-robot G1 station")
    parser.add_argument("--allow-motion", action="store_true")
    parser.add_argument(
        "--engage", action="store_true", help="G1 only: hand SONIC its start after startup"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8222)
    parser.add_argument(
        "--viser-port", type=int, default=8080, help="Camera viewer and Home button"
    )
    parser.add_argument("--no-viser", action="store_true", help="Disable the browser visualizer")
    parser.add_argument("--record-dir", type=Path)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--sam3-url")
    parser.add_argument("--curobo-url")
    args = parser.parse_args(argv)
    g1 = args.benchmark == "g1_real"
    if args.engage and not args.allow_motion:
        error = "--engage requires --allow-motion"
    elif (args.engage or args.fake) and not g1:
        error = "--engage and --fake apply to --benchmark g1_real only"
    elif args.sim and g1:
        error = "--sim applies to --benchmark yam_real only; use --fake"
    else:
        error = None
    if error:
        print(error, file=sys.stderr)
        return 2
    from contextlib import ExitStack

    import uvicorn

    from cap_harness.providers.yam_kinematics import YamKinematicsIKProvider
    from cap_harness.providers.yam_kinematics.client import ARM_DOF

    adapter_type, public_tool_names, real_station, offline_station, kinematics_type = (
        _real_embodiment(args.benchmark)
    )
    with ExitStack() as stack:
        factory = offline_station if (args.sim or args.fake) else real_station
        env = factory(args.station, config_root=args.station_config_root)
        stack.callback(env.close)
        episode = None
        event_sink = tool_sink = None
        if args.record_dir:
            from cap_harness.yam_real.recorder import YamEpisodeRecorder

            args.record_dir.mkdir(parents=True, exist_ok=False)
            episode = YamEpisodeRecorder(
                args.record_dir / "episode",
                video_dir=args.record_dir / "videos",
                fps=round(env.config.controller.control_frequency_hz),
                cameras=() if args.no_video else None,
                task="system2",
            )
            episode.start(
                env, meta={"station": args.station, "physical_motion_authorized": args.allow_motion}
            )
            stack.callback(episode.finalize, terminal_event="session_closed")
            log_lock = threading.Lock()

            def record(path: str, value: Any) -> None:
                with log_lock, (args.record_dir / path).open("a") as stream:
                    stream.write(json.dumps(value, default=str) + "\n")

            event_sink = lambda event: record("events.jsonl", asdict(event))
            tool_sink = lambda **result: record("tools.jsonl", result)
        options: dict[str, Any] = {
            "ik_providers": {
                "mink": YamKinematicsIKProvider(
                    kinematics_type(env.config.model_xml), dof=getattr(env, "dof", ARM_DOF)
                )
            }
        }
        if args.sam3_url:
            from cap_harness.providers.sam3 import Sam3Provider

            options["segmentation_provider"] = Sam3Provider(base_url=args.sam3_url)
        if args.curobo_url:
            from cap_harness.providers.curobo import CuRoboProvider

            planner = CuRoboProvider(
                base_url=args.curobo_url,
                control_period_s=1 / env.config.controller.control_frequency_hz,
            )
            options["ik_providers"]["curobo"] = planner
            options["trajectory_planning_providers"] = {"curobo": planner}
            options["integrated_pose_planning_providers"] = {"curobo-integrated": planner}
        adapter, runtime = build_runtime(
            env,
            adapter_type=adapter_type,
            public_tool_names=public_tool_names,
            family=args.benchmark,
            allow_motion=args.allow_motion,
            episode_recorder=episode,
            event_sink=event_sink,
            tool_result_sink=tool_sink,
            **options,
        )
        # Stop jobs before finalizing recordings, then release cameras and arms.
        stack.callback(runtime.close)
        if args.engage:
            try:
                print(f"engage: {env.engage()}")
            except RuntimeError as exc:
                print(f"engage refused: {exc}", file=sys.stderr)
                return 1
        if not args.no_viser:
            from .visualizer import System2Visualizer

            visualizer = System2Visualizer(
                runtime.session,
                allow_motion=args.allow_motion,
                host=args.host,
                port=args.viser_port,
                # No Viser model for G1 yet: cameras, status, and buttons only.
                **({"urdf_path": None, "engage": env.engage} if g1 else {}),
            )
            stack.callback(visualizer.close)
        uvicorn.run(create_app(runtime.session), host=args.host, port=args.port)
    return 0

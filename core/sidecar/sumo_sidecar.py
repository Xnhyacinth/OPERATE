"""
core.sidecar.sumo_sidecar — SUMO transport probe + orphan-safe lifecycle.

This is the ONLY module that knows how to reach a SUMO process. It chooses a
transport at probe time and exposes a tiny, backend-agnostic surface:

    sidecar = SumoSidecar(net_path, route_path, ...)
    if sidecar.available():          # cheap; no process launched
        sidecar.start()              # launches SUMO via the chosen transport
        snap = sidecar.snapshot()    # one cached pull per tick
        sidecar.simulation_step(n)   # advance n deterministic substeps
        sidecar.close()              # finally-guarded; force-kills orphans

Transport selection order (``docs/v0.7_traffic_spec.md`` §2/§11):

1. ``libsumo`` — in-process Python binding (canonical replay transport,
   15–30× faster on micro nets, no socket round-trip).
2. ``traci`` — TCP to a spawned ``sumo`` binary (Docker-CI parity path).
3. Docker ``eclipse/sumo`` — pinned image, ``traci`` over a published port,
   only when neither Python binding nor a native ``sumo`` binary is present.

Determinism flags pinned at start (``docs/v0.7_traffic_spec.md`` §9):
``--seed``, ``--step-length``, ``--default.action-step-length``,
``--routing-algorithm``, and crucially ``--time-to-teleport=-1`` (default 300 s
teleport would silently swallow incident-clearance metrics; stuck vehicles must
surface as a ``safety_violation`` signal instead).

**No ``core`` backend import.** This module imports only the stdlib plus the
optional SUMO Python libraries (probed lazily). It never imports ``domains.*``.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.metadata
import importlib.util
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Transport = Literal["libsumo", "traci", "docker"]


class SumoSidecarUnavailable(RuntimeError):
    """Raised by :meth:`SumoSidecar.start` when no SUMO transport is reachable.

    Constructible-but-unavailable is intentional: stage 1–3 (mock-only) hosts
    build the sidecar, see ``available() is False``, and skip live execution
    without ever raising. Only an explicit ``start()`` on such a host raises.
    """


def probe_sumo_transport() -> Transport | None:
    """Return the best available SUMO transport, or ``None`` if SUMO is absent.

    Pure detection — launches nothing. Mirrors ``audit._egret_available()`` /
    ``_runtime_unavailable()``: ``importlib.util.find_spec`` for the Python
    bindings, ``shutil.which`` for the native binary, ``docker`` CLI for the
    container fallback. The ``OPERATE_TRAFFIC_FORCE_TRANSPORT`` env var pins a
    specific transport for reproducibility (release transport-pinning, §9).
    """
    forced = os.environ.get("OPERATE_TRAFFIC_FORCE_TRANSPORT")
    if forced in ("libsumo", "traci", "docker"):
        return forced  # type: ignore[return-value]
    # Ensure sumo/tools (traci) is on sys.path for eclipse-sumo pip package
    _sumo_spec = importlib.util.find_spec("sumo")
    _sumo_root = None
    if _sumo_spec is not None and _sumo_spec.submodule_search_locations:
        _sumo_root = Path(_sumo_spec.submodule_search_locations[0])
        _sumo_tools = _sumo_root / "tools"
        if _sumo_tools.is_dir() and str(_sumo_tools) not in sys.path:
            sys.path.insert(0, str(_sumo_tools))

    if importlib.util.find_spec("libsumo") is not None:
        return "libsumo"
    # traci as a TCP client needs both the python lib AND a sumo binary to spawn.
    # eclipse-sumo pip package places sumo in <package_root>/bin/ — add it to
    # the PATH probe so shutil.which("sumo") finds it without global install.
    _traci_found = importlib.util.find_spec("traci") is not None
    _sumo_bin = shutil.which("sumo")
    if _sumo_root is not None and _sumo_bin is None:
        _candidate = str(_sumo_root / "bin" / "sumo")
        if os.path.isfile(_candidate) and os.access(_candidate, os.X_OK):
            _sumo_bin = _candidate
    if _traci_found and _sumo_bin is not None:
        return "traci"
    # Docker fallback: traci lib present + a usable docker CLI + an opt-in flag,
    # because pulling/launching a container is heavy and must be explicit.
    if (
        (importlib.util.find_spec("traci") is not None or _find_spec_safe("sumo.tools.traci") is not None)
        and shutil.which("docker") is not None
        and os.environ.get("OPERATE_TRAFFIC_ALLOW_DOCKER") == "1"
    ):
        return "docker"
    return None


def _find_spec_safe(dotted_name: str):
    """``importlib.util.find_spec`` for a dotted submodule (e.g.
    ``sumo.tools.traci``) raises ``ModuleNotFoundError`` — rather than
    returning ``None`` — when an intermediate package in the path (here
    ``sumo``) isn't importable at all. On a machine without the
    ``eclipse-sumo`` pip package this crashed :func:`probe_sumo_transport`
    outright, breaking the graceful-skip contract every other gated backend
    (EGRET AC-OPF, RCRS) relies on. Treat "parent package missing" the same
    as "submodule not found"."""
    try:
        return importlib.util.find_spec(dotted_name)
    except ModuleNotFoundError:
        return None


def _resolve_traci_launch() -> tuple[str, dict[str, str] | None]:
    """Bypass only the installed wheel's subprocess launcher, not system SUMO."""
    selected = shutil.which("sumo")
    try:
        distribution = importlib.metadata.distribution("eclipse-sumo")
    except importlib.metadata.PackageNotFoundError:
        distribution = None
    package_root: Path | None = None
    if distribution is not None:
        wheel_launcher = bool(selected) and any(
            entry.group == "console_scripts"
            and entry.name == "sumo" and entry.value == "sumo:sumo"
            for entry in distribution.entry_points
        ) and any(
            str(record) != "sumo/bin/sumo"
            and record.name == "sumo"
            and Path(distribution.locate_file(record)).resolve() == Path(selected).resolve()
            for record in distribution.files or []
        )
        if selected is None or wheel_launcher:
            package_root = Path(distribution.locate_file("sumo"))
    elif selected is None:
        spec = _find_spec_safe("sumo")
        if spec is not None and spec.submodule_search_locations:
            package_root = Path(spec.submodule_search_locations[0])
    if package_root is not None:
        native = package_root / "bin" / "sumo"
        if not native.is_file() or not os.access(native, os.X_OK):
            raise SumoSidecarUnavailable(f"SUMO wheel native binary unavailable: {native}")
        # Mirror eclipse-sumo's __init__ environment setup without spawning its
        # Python console script, so Popen owns the actual simulator on failure.
        environment = dict(os.environ)
        environment.setdefault("SUMO_HOME", str(package_root))
        if not environment.get("PROJ_LIB") and not environment.get("PROJ_DATA"):
            environment["PROJ_LIB"] = environment["PROJ_DATA"] = str(package_root / "data" / "proj")
        return str(native), environment
    return selected or "sumo", None


def sumo_available() -> bool:
    """Convenience boolean for audit graceful-skip + ``--mock-only`` guards."""
    return probe_sumo_transport() is not None


@contextmanager
def _sumo_start_port_lock():
    """Serialize dynamic TraCI port allocation through the initial handshake.

    ``traci.getFreeSocketPort()`` closes its probe socket before SUMO binds the
    returned port.  Concurrent benchmark workers can therefore select the same
    port and attach to the wrong child.  A host-local file lock closes that
    time-of-check/time-of-use window without serializing the simulations.
    """

    lock_path = Path(tempfile.gettempdir()) / "operate-sumo-traci-start.lock"
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":  # pragma: no cover - exercised on Windows hosts
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _optional_finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _call_optional_float(target: Any, name: str, *args: Any) -> float | None:
    getter = getattr(target, name, None)
    if not callable(getter):
        return None
    try:
        return _optional_finite(getter(*args))
    except Exception:
        return None


@dataclass
class SumoSidecar:
    """Owns one SUMO process behind a probed transport.

    The sidecar is deliberately thin: it spawns/attaches, advances substeps,
    and pulls a *single* cached snapshot per benchmark tick (within-tick fog
    purity). All physics interpretation lives in
    ``domains/traffic/backends/sumo_backend.py``; the sidecar only moves bytes.
    """

    net_path: str
    route_path: str
    seed: int = 0
    step_length: float = 1.0
    routing_algorithm: str = "dijkstra"
    docker_image: str = "eclipse/sumo:1.20.0"
    # ``None`` delegates to traci.getFreeSocketPort(). A fixed default port
    # makes otherwise independent worker processes collide under parallel
    # replay.
    traci_port: int | None = None
    extra_args: tuple[str, ...] = ()
    config_path: str | None = None

    _transport: Transport | None = field(default=None, init=False)
    _conn: Any = field(default=None, init=False, repr=False)
    _proc: subprocess.Popen[bytes] | None = field(
        default=None, init=False, repr=False
    )
    _traci_label: str | None = field(default=None, init=False, repr=False)
    _started: bool = field(default=False, init=False)
    _snapshot_cache: dict[str, Any] | None = field(
        default=None, init=False, repr=False
    )
    _snapshot_tick: int = field(default=-1, init=False)
    _edge_lane_ids_cache: dict[str, tuple[str, ...]] | None = field(
        default=None, init=False, repr=False
    )
    _lane_edge_id_cache: dict[str, str] | None = field(
        default=None, init=False, repr=False
    )

    # ── probe ───────────────────────────────────────────────────────────────

    def available(self) -> bool:
        """True if a transport could be selected without launching anything."""
        if self._transport is None:
            self._transport = probe_sumo_transport()
        return self._transport is not None

    @property
    def transport(self) -> Transport | None:
        if self._transport is None:
            self._transport = probe_sumo_transport()
        return self._transport

    @property
    def connection(self) -> Any:
        """Return the active libsumo/TraCI connection for domain adapters.

        The sidecar still owns process lifecycle and transport selection; this
        narrow read-only handle lets a domain adapter use native SUMO APIs
        without duplicating subprocess/TraCI startup code.
        """
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("connection requested before start()")
        return self._conn

    def runtime_version(self) -> str:
        """Return the connected SUMO server version used by this episode."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("runtime_version() before start()")
        getter = getattr(self._conn, "getVersion", None)
        if callable(getter):
            raw = getter()
            if isinstance(raw, (tuple, list)) and raw:
                return " ".join(str(value) for value in raw)
            return str(raw)
        return "unknown"

    # ── deterministic launch args ────────────────────────────────────────────

    def _sumo_cmd(self) -> list[str]:
        """Assemble the deterministic SUMO CLI (shared by traci/docker)."""
        if self.config_path:
            return [
                "-c",
                self.config_path,
                "--seed",
                str(int(self.seed)),
                "--step-length",
                str(float(self.step_length)),
                "--default.action-step-length",
                str(float(self.step_length)),
                "--routing-algorithm",
                self.routing_algorithm,
                "--time-to-teleport",
                "-1",
                "--no-step-log",
                "true",
                "--no-warnings",
                "true",
                *self.extra_args,
                # ``sumo`` is headless, but still opens a gui-settings file
                # named by a config unless an empty override is explicit.
                "--gui-settings-file",
                "",
            ]
        return [
            "-n",
            self.net_path,
            "-r",
            self.route_path,
            "--seed",
            str(int(self.seed)),
            "--step-length",
            str(float(self.step_length)),
            "--default.action-step-length",
            str(float(self.step_length)),
            "--routing-algorithm",
            self.routing_algorithm,
            # §9: never teleport — stuck vehicles are a safety signal, not noise.
            "--time-to-teleport",
            "-1",
            "--no-step-log",
            "true",
            "--no-warnings",
            "true",
            *self.extra_args,
        ]

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> Transport:
        """Launch / attach SUMO via the probed transport. Raises if unavailable."""
        if self._started:
            return self._transport  # type: ignore[return-value]
        if Path(self.net_path).is_file():
            self.prime_topology_cache()
        transport = self.transport
        if transport is None:
            raise SumoSidecarUnavailable(
                "No SUMO transport reachable (need importable 'libsumo', or "
                "'traci' + a 'sumo' binary, or 'traci' + docker with "
                "OPERATE_TRAFFIC_ALLOW_DOCKER=1). Run stage 1–3 with --mock-only."
            )
        try:
            if transport == "libsumo":
                self._start_libsumo()
            elif transport == "traci":
                self._start_traci()
            else:
                self._start_docker()
        except Exception:
            # Launch is all-or-nothing: never leave a half-open connection or
            # orphan process behind if any step of start() throws.
            self.close()
            raise
        self._started = True
        return transport

    def prime_topology_cache(self) -> None:
        """Cache static edge/lane identities from the exact network asset."""
        if self._edge_lane_ids_cache is not None:
            return
        path = Path(self.net_path)
        opener = gzip.open if path.suffix == ".gz" else open
        edge_lanes: dict[str, tuple[str, ...]] = {}
        lane_edges: dict[str, str] = {}
        with opener(path, "rb") as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag.rsplit("}", 1)[-1] != "edge":
                    continue
                edge_id = str(element.attrib.get("id") or "").strip()
                lane_ids = tuple(
                    str(child.attrib.get("id") or "").strip()
                    for child in element
                    if child.tag.rsplit("}", 1)[-1] == "lane"
                    and str(child.attrib.get("id") or "").strip()
                )
                if edge_id and lane_ids:
                    edge_lanes[edge_id] = lane_ids
                    lane_edges.update({lane_id: edge_id for lane_id in lane_ids})
                element.clear()
        if not edge_lanes:
            raise ValueError(f"SUMO network contains no edge/lane topology: {path}")
        self._edge_lane_ids_cache = edge_lanes
        self._lane_edge_id_cache = lane_edges

    def _start_libsumo(self) -> None:
        import libsumo  # type: ignore[import-not-found]

        libsumo.start(["sumo", *self._sumo_cmd()])
        self._conn = libsumo

    def _start_traci(self) -> None:
        import traci  # type: ignore[import-not-found]

        sumo_bin, environment = _resolve_traci_launch()
        start_lock = (
            _sumo_start_port_lock() if self.traci_port is None else nullcontext()
        )
        with start_lock:
            port = self.traci_port
            if port is None:
                port = int(traci.getFreeSocketPort())
            label = f"sumo-sidecar-{os.getpid()}-{id(self)}"
            command = [*self._sumo_cmd(), "--remote-port", str(port)]
            # Do not use ``traci.start`` here.  Its implementation creates the
            # child internally and leaves it orphaned when the initial handshake
            # fails before a labelled connection is registered.  Owning the Popen
            # object lets ``start()``'s failure path and ``close()`` terminate the
            # exact process that belongs to this sidecar.
            self._proc = subprocess.Popen(
                [sumo_bin, *command],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=environment,
            )
            try:
                retries = max(
                    0,
                    int(os.environ.get("OPERATE_SUMO_CONNECT_RETRIES", "30")),
                )
                wait_seconds = max(
                    0.05,
                    float(
                        os.environ.get(
                            "OPERATE_SUMO_CONNECT_RETRY_SECONDS", "0.5"
                        )
                    ),
                )
                connection = traci.connect(
                    port=port,
                    numRetries=retries,
                    host="localhost",
                    proc=self._proc,
                    waitBetweenRetries=wait_seconds,
                    label=label,
                )
                traci.switch(label)
                self._traci_label = label
                self._conn = connection
            except Exception:
                # Keep the lock until the failed child releases its port.
                self.close()
                raise

    def _start_docker(self) -> None:
        import traci  # type: ignore[import-not-found]

        # Launch the pinned image, publishing the TraCI port, then attach.
        port = self.traci_port or traci.getFreeSocketPort()
        self._proc = subprocess.Popen(
            [
                "docker",
                "run",
                "--rm",
                "-p",
                f"{port}:{port}",
                self.docker_image,
                "sumo",
                "--remote-port",
                str(port),
                *self._sumo_cmd(),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        traci.init(port=port)
        self._conn = traci

    def simulation_step(self, substeps: int = 1) -> None:
        """Advance SUMO by ``substeps`` deterministic substeps.

        Invalidates the per-tick snapshot cache so the next :meth:`snapshot`
        pulls fresh state exactly once.
        """
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("simulation_step() before start()")
        for _ in range(max(1, int(substeps))):
            self._conn.simulationStep()
        self._snapshot_cache = None

    def set_traffic_light_program(self, tls_id: str, program_id: str) -> dict[str, Any]:
        """Set a SUMO traffic-light program and read back the applied state."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("set_traffic_light_program() before start()")
        trafficlight = getattr(self._conn, "trafficlight", None)
        setter = getattr(trafficlight, "setProgram", None)
        if not callable(setter):
            raise SumoSidecarUnavailable(
                "SUMO trafficlight.setProgram is not available"
            )
        tls = str(tls_id)
        program = str(program_id)
        setter(tls, program)
        readback: str | None = None
        getter = getattr(trafficlight, "getProgram", None)
        if callable(getter):
            readback = str(getter(tls))
        self._snapshot_cache = None
        return {
            "sumo_tls_id": tls,
            "sumo_program_id": program,
            "sumo_program_readback": readback,
            "sumo_program_readback_available": readback is not None,
            "sumo_program_readback_matches": (
                readback == program if readback is not None else None
            ),
            "sumo_state_mutated": True,
        }

    def set_traffic_light_phase_duration(
        self, tls_id: str, phase_duration_s: float
    ) -> dict[str, Any]:
        """Override the current TLS phase duration and read back phase state.

        This is deliberately narrower than a full program-logic editor: it uses
        SUMO's native ``trafficlight.setPhaseDuration`` on the current phase so a
        live probe can derive a state-changing action from observed queues
        without fabricating a new timing plan. Domain code decides when this is a
        meaningful control; the sidecar only mutates and reports SUMO state.
        """
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable(
                "set_traffic_light_phase_duration() before start()"
            )
        trafficlight = getattr(self._conn, "trafficlight", None)
        setter = getattr(trafficlight, "setPhaseDuration", None)
        if not callable(setter):
            raise SumoSidecarUnavailable(
                "SUMO trafficlight.setPhaseDuration is not available"
            )
        tls = str(tls_id)
        duration = float(phase_duration_s)
        setter(tls, duration)
        phase_getter = getattr(trafficlight, "getPhase", None)
        next_switch_getter = getattr(trafficlight, "getNextSwitch", None)
        spent_getter = getattr(trafficlight, "getSpentDuration", None)
        self._snapshot_cache = None
        return {
            "sumo_tls_id": tls,
            "sumo_phase_duration_s": duration,
            "sumo_phase_index": int(phase_getter(tls))
            if callable(phase_getter)
            else None,
            "sumo_next_switch_s": (
                float(next_switch_getter(tls)) if callable(next_switch_getter) else None
            ),
            "sumo_spent_duration_s": (
                float(spent_getter(tls)) if callable(spent_getter) else None
            ),
            "sumo_state_mutated": True,
        }

    def controlled_lanes(self, tls_id: str) -> tuple[str, ...]:
        """Return the de-duplicated lanes a traffic-light controls (order-stable).

        Generic TraCI surface (``trafficlight.getControlledLanes``); the sidecar
        stays domain-agnostic — the caller decides which TLS belongs to which
        benchmark corridor. SUMO repeats a lane once per signal phase it appears
        in, so duplicates are collapsed while preserving first-seen order.
        """
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("controlled_lanes() before start()")
        getter = getattr(
            getattr(self._conn, "trafficlight", None), "getControlledLanes", None
        )
        if not callable(getter):
            raise SumoSidecarUnavailable(
                "SUMO trafficlight.getControlledLanes is not available"
            )
        seen: dict[str, None] = {}
        for lane in getter(str(tls_id)):
            seen.setdefault(str(lane), None)
        return tuple(seen.keys())

    def traffic_light_program_ids(self, tls_id: str) -> tuple[str, ...]:
        """Return program ids actually declared by a live SUMO TLS."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable(
                "traffic_light_program_ids() before start()"
            )
        getter = getattr(
            getattr(self._conn, "trafficlight", None),
            "getAllProgramLogics",
            None,
        )
        if not callable(getter):
            raise SumoSidecarUnavailable(
                "SUMO trafficlight.getAllProgramLogics is not available"
            )
        return tuple(
            dict.fromkeys(
                str(getattr(logic, "programID", ""))
                for logic in getter(str(tls_id))
                if str(getattr(logic, "programID", ""))
            )
        )

    def traffic_light_ids(self) -> tuple[str, ...]:
        """Return exact runtime TLS identifiers in stable lexical order."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("traffic_light_ids() before start()")
        getter = getattr(
            getattr(self._conn, "trafficlight", None), "getIDList", None
        )
        if not callable(getter):
            raise SumoSidecarUnavailable(
                "SUMO trafficlight.getIDList is not available"
            )
        return tuple(sorted(str(value) for value in getter()))

    def controlled_links(self, tls_id: str) -> tuple[tuple[dict[str, str], ...], ...]:
        """Return exact incoming/outgoing/via lane identities for one TLS."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("controlled_links() before start()")
        getter = getattr(
            getattr(self._conn, "trafficlight", None), "getControlledLinks", None
        )
        if not callable(getter):
            raise SumoSidecarUnavailable(
                "SUMO trafficlight.getControlledLinks is not available"
            )
        groups: list[tuple[dict[str, str], ...]] = []
        for group in getter(str(tls_id)):
            normalized: list[dict[str, str]] = []
            for link in group or ():
                values = list(link)
                normalized.append(
                    {
                        "incoming_lane": str(values[0]) if values else "",
                        "outgoing_lane": str(values[1]) if len(values) > 1 else "",
                        "via_lane": str(values[2]) if len(values) > 2 else "",
                    }
                )
            groups.append(tuple(normalized))
        return tuple(groups)

    def vehicle_control_context(self, tls_id: str) -> list[dict[str, Any]]:
        """Capture source-lineage and native signal context for controlled vehicles."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable(
                "vehicle_control_context() before start()"
            )
        lane = getattr(self._conn, "lane", None)
        vehicle = getattr(self._conn, "vehicle", None)
        trafficlight = getattr(self._conn, "trafficlight", None)
        lane_vehicle_ids = getattr(lane, "getLastStepVehicleIDs", None)
        road_id = getattr(vehicle, "getRoadID", None)
        lane_id = getattr(vehicle, "getLaneID", None)
        route_id = getattr(vehicle, "getRouteID", None)
        route = getattr(vehicle, "getRoute", None)
        if not all(
            callable(getter)
            for getter in (
                lane_vehicle_ids,
                road_id,
                lane_id,
                route_id,
                route,
            )
        ):
            raise SumoSidecarUnavailable(
                "SUMO vehicle/lane lineage API is not available"
            )
        tls = str(tls_id)
        program = str(trafficlight.getProgram(tls))
        phase_index = int(trafficlight.getPhase(tls))
        phase_state = str(trafficlight.getRedYellowGreenState(tls))
        simulation_time = float(self._conn.simulation.getTime())
        captured: dict[str, dict[str, Any]] = {}
        for link_index, group in enumerate(self.controlled_links(tls)):
            for link in group:
                incoming_lane = str(link.get("incoming_lane") or "")
                for raw_vehicle_id in lane_vehicle_ids(incoming_lane):
                    runtime_vehicle_id = str(raw_vehicle_id)
                    if runtime_vehicle_id in captured:
                        continue
                    runtime_route_id = str(route_id(runtime_vehicle_id))
                    edge_sequence = [
                        str(value) for value in route(runtime_vehicle_id)
                    ]
                    captured[runtime_vehicle_id] = {
                        "vehicle_id": runtime_vehicle_id,
                        "simulation_time": simulation_time,
                        "edge_id": str(road_id(runtime_vehicle_id)),
                        "lane_id": str(lane_id(runtime_vehicle_id)),
                        "route_id": runtime_route_id,
                        "trip_id": runtime_vehicle_id,
                        "source_event_ids": [runtime_vehicle_id],
                        "tls_context": {"tls_id": tls},
                        "controlled_link_context": {
                            "link_index": link_index,
                            **link,
                        },
                        "phase_context": {
                            "program_id": program,
                            "phase_index": phase_index,
                            "phase_state": phase_state,
                            "link_signal_state": (
                                phase_state[link_index]
                                if link_index < len(phase_state)
                                else ""
                            ),
                        },
                        "source_lineage": {
                            "source_trip_id": runtime_vehicle_id,
                            "runtime_vehicle_id": runtime_vehicle_id,
                            "event_vehicle_id": runtime_vehicle_id,
                            "route_id": runtime_route_id,
                            "edge_sequence": edge_sequence,
                        },
                    }
        return [captured[key] for key in sorted(captured)]

    def traffic_light_contract(self, tls_id: str) -> dict[str, Any]:
        """Read the full native program/phase contract for one exact TLS."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable(
                "traffic_light_contract() before start()"
            )
        trafficlight = getattr(self._conn, "trafficlight", None)
        logic_getter = getattr(trafficlight, "getAllProgramLogics", None)
        if not callable(logic_getter):
            raise SumoSidecarUnavailable(
                "SUMO trafficlight.getAllProgramLogics is not available"
            )
        tls = str(tls_id)
        programs: dict[str, Any] = {}
        for logic in logic_getter(tls):
            program_id = str(getattr(logic, "programID", ""))
            if not program_id:
                continue
            phases = []
            for index, phase in enumerate(getattr(logic, "phases", ()) or ()):
                next_values = getattr(phase, "next", ()) or ()
                phases.append(
                    {
                        "index": index,
                        "duration": float(getattr(phase, "duration", 0.0)),
                        "min_duration": _optional_finite(
                            getattr(phase, "minDur", None)
                        ),
                        "max_duration": _optional_finite(
                            getattr(phase, "maxDur", None)
                        ),
                        "state": str(getattr(phase, "state", "")),
                        "next": [int(value) for value in next_values],
                        "name": _optional_text(getattr(phase, "name", None)),
                    }
                )
            programs[program_id] = {"phases": phases}
        current_program = str(trafficlight.getProgram(tls))
        current_phase = int(trafficlight.getPhase(tls))
        spent = _call_optional_float(trafficlight, "getSpentDuration", tls)
        next_switch = _call_optional_float(trafficlight, "getNextSwitch", tls)
        sim_time = float(self._conn.simulation.getTime())
        phase = (programs.get(current_program) or {}).get("phases", [])
        current_phase_row = (
            dict(phase[current_phase])
            if 0 <= current_phase < len(phase)
            else {}
        )
        return {
            "tls_id": tls,
            "current_program": current_program,
            "current_phase": current_phase,
            "current_state": str(trafficlight.getRedYellowGreenState(tls)),
            "sim_time": sim_time,
            "spent_duration": spent,
            "next_switch": next_switch,
            "remaining_duration": (
                max(0.0, next_switch - sim_time)
                if next_switch is not None
                else None
            ),
            "current_phase_bounds": {
                "min_duration": current_phase_row.get("min_duration"),
                "max_duration": current_phase_row.get("max_duration"),
            },
            "controlled_lanes": list(self.controlled_lanes(tls)),
            "controlled_links": [
                list(group) for group in self.controlled_links(tls)
            ],
            "programs": programs,
        }

    def traffic_light_runtime_state(self, tls_id: str) -> dict[str, Any]:
        """Read current TLS state without reloading its full logic inventory."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable(
                "traffic_light_runtime_state() before start()"
            )
        trafficlight = getattr(self._conn, "trafficlight", None)
        tls = str(tls_id)
        sim_time = float(self._conn.simulation.getTime())
        next_switch = _call_optional_float(
            trafficlight, "getNextSwitch", tls
        )
        spent = _call_optional_float(
            trafficlight, "getSpentDuration", tls
        )
        return {
            "tls_id": tls,
            "current_program": str(trafficlight.getProgram(tls)),
            "current_phase": int(trafficlight.getPhase(tls)),
            "current_state": str(
                trafficlight.getRedYellowGreenState(tls)
            ),
            "spent_duration": spent,
            "next_switch": next_switch,
            "remaining_duration": (
                max(0.0, next_switch - sim_time)
                if next_switch is not None
                else None
            ),
            "sim_time": sim_time,
        }

    def lane_group_metrics(self, lane_ids: tuple[str, ...]) -> dict[str, Any]:
        """Aggregate live SUMO metrics over a lane group (one real query/tick).

        Sums per-lane ``getLastStepHaltingNumber`` (queued vehicles),
        ``getLastStepVehicleNumber`` (occupancy), and ``getWaitingTime``
        (accumulated waiting seconds) across ``lane_ids``. Generic TraCI surface
        (``conn.lane.*``); physics interpretation (delay minutes, equity) is the
        backend's job. Returns zeros for an empty group rather than raising, so a
        corridor whose bound TLS controls no lanes is honestly empty, not an error.
        """
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("lane_group_metrics() before start()")
        lane = getattr(self._conn, "lane", None)
        halting_getter = getattr(lane, "getLastStepHaltingNumber", None)
        vehicle_getter = getattr(lane, "getLastStepVehicleNumber", None)
        waiting_getter = getattr(lane, "getWaitingTime", None)
        speed_getter = getattr(lane, "getLastStepMeanSpeed", None)
        occupancy_getter = getattr(lane, "getLastStepOccupancy", None)
        if not (
            callable(halting_getter)
            and callable(vehicle_getter)
            and callable(waiting_getter)
        ):
            raise SumoSidecarUnavailable("SUMO lane.* metric API is not available")
        halting = 0.0
        vehicles = 0.0
        waiting_s = 0.0
        mean_speed_sum = 0.0
        occupancy_sum = 0.0
        for lane_id in lane_ids:
            lid = str(lane_id)
            halting += float(halting_getter(lid))
            vehicles += float(vehicle_getter(lid))
            waiting_s += float(waiting_getter(lid))
            if callable(speed_getter):
                mean_speed_sum += float(speed_getter(lid))
            if callable(occupancy_getter):
                occupancy_sum += float(occupancy_getter(lid))
        return {
            "n_lanes": len(lane_ids),
            "halting": halting,
            "vehicles": vehicles,
            "waiting_time_s": waiting_s,
            "mean_speed_mps": (
                mean_speed_sum / len(lane_ids) if lane_ids else 0.0
            ),
            "mean_occupancy_percent": (
                occupancy_sum / len(lane_ids) if lane_ids else 0.0
            ),
        }

    def network_counts(self) -> dict[str, int | None]:
        """Return network-wide lane/edge denominators when SUMO exposes them."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("network_counts() before start()")
        if Path(self.net_path).is_file():
            self.prime_topology_cache()
            return {
                "n_lanes": len(self._lane_edge_id_cache or {}),
                "n_edges": len(self._edge_lane_ids_cache or {}),
            }
        lane_getter = getattr(getattr(self._conn, "lane", None), "getIDList", None)
        edge_getter = getattr(getattr(self._conn, "edge", None), "getIDList", None)
        return {
            "n_lanes": len(lane_getter()) if callable(lane_getter) else None,
            "n_edges": len(edge_getter()) if callable(edge_getter) else None,
        }

    def simulation_counts(self) -> dict[str, int | float | None]:
        """Read per-step vehicle counts without enumerating the full network."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("simulation_counts() before start()")
        minimum_expected_getter = getattr(
            self._conn.simulation,
            "getMinExpectedNumber",
            None,
        )
        return {
            "n_vehicles": int(self._conn.vehicle.getIDCount()),
            "sim_time": float(self._conn.simulation.getTime()),
            "arrived": int(self._conn.simulation.getArrivedNumber()),
            "departed": int(self._conn.simulation.getDepartedNumber()),
            "minimum_expected": (
                int(minimum_expected_getter())
                if callable(minimum_expected_getter)
                else None
            ),
        }

    def route_ids(self) -> tuple[str, ...]:
        """Return route identifiers loaded by the active SUMO asset graph.

        The sidecar exposes only the generic TraCI route inventory.  Scenario
        code must choose one of these runtime identifiers; it may not invent a
        route name from YAML.  This is used by deterministic procedural traffic
        events that inject demand while keeping the route definition source
        grounded in the locked ``.rou.xml``/``.sumocfg`` graph.
        """
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("route_ids() before start()")
        getter = getattr(getattr(self._conn, "route", None), "getIDList", None)
        if not callable(getter):
            raise SumoSidecarUnavailable("SUMO route.getIDList is not available")
        return tuple(sorted({str(value) for value in getter() if str(value)}))

    def edge_ids(self) -> tuple[str, ...]:
        """Return edge identifiers parsed from the exact loaded network asset."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("edge_ids() before start()")
        if Path(self.net_path).is_file():
            self.prime_topology_cache()
            return tuple(sorted(self._edge_lane_ids_cache or {}))
        getter = getattr(getattr(self._conn, "edge", None), "getIDList", None)
        if not callable(getter):
            raise SumoSidecarUnavailable("SUMO edge.getIDList is not available")
        return tuple(sorted({str(value) for value in getter() if str(value)}))

    def edge_lane_ids(self, edge_id: str) -> tuple[str, ...]:
        """Return cached lane identities for one exact network edge."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("edge_lane_ids() before start()")
        normalized = str(edge_id).strip()
        if Path(self.net_path).is_file():
            self.prime_topology_cache()
        lanes = (self._edge_lane_ids_cache or {}).get(normalized)
        if lanes is None and self._edge_lane_ids_cache is None:
            lane_api = getattr(self._conn, "lane", None)
            id_getter = getattr(lane_api, "getIDList", None)
            edge_getter = getattr(lane_api, "getEdgeID", None)
            if callable(id_getter) and callable(edge_getter):
                lanes = tuple(
                    sorted(
                        str(value)
                        for value in id_getter()
                        if str(edge_getter(str(value))) == normalized
                    )
                )
        if not lanes:
            raise ValueError(
                f"edge_id is not present in the runtime graph: {edge_id}"
            )
        return lanes

    def lane_edge_id(self, lane_id: str) -> str:
        """Return the cached parent edge for one exact network lane."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("lane_edge_id() before start()")
        normalized = str(lane_id).strip()
        if Path(self.net_path).is_file():
            self.prime_topology_cache()
        edge_id = (self._lane_edge_id_cache or {}).get(normalized)
        if edge_id is None and self._lane_edge_id_cache is None:
            getter = getattr(getattr(self._conn, "lane", None), "getEdgeID", None)
            if callable(getter):
                edge_id = str(getter(normalized))
        if edge_id is None:
            raise ValueError(
                f"lane_id is not present in the runtime graph: {lane_id}"
            )
        return edge_id

    def edge_lane_max_speeds(self, edge_id: str) -> dict[str, float]:
        """Read every lane speed limit for one runtime-validated edge."""
        edge_id = str(edge_id).strip()
        lane_ids = self.edge_lane_ids(edge_id)
        lane_api = getattr(self._conn, "lane", None)
        speed_getter = getattr(lane_api, "getMaxSpeed", None)
        if not callable(speed_getter):
            raise SumoSidecarUnavailable(
                "SUMO lane max-speed API is not available"
            )
        speeds = {lane_id: float(speed_getter(lane_id)) for lane_id in lane_ids}
        if any(not math.isfinite(value) or value <= 0.0 for value in speeds.values()):
            raise RuntimeError(f"runtime edge has invalid lane max speed: {edge_id}")
        return speeds

    def set_edge_lane_max_speeds(
        self,
        *,
        edge_id: str,
        lane_max_speeds: dict[str, float],
    ) -> dict[str, Any]:
        """Apply and verify native lane speed limits for one runtime edge."""
        before = self.edge_lane_max_speeds(edge_id)
        requested = {str(key): float(value) for key, value in lane_max_speeds.items()}
        if set(requested) != set(before):
            raise ValueError("lane_max_speeds must cover exactly the runtime edge lanes")
        if any(not math.isfinite(value) or value <= 0.0 for value in requested.values()):
            raise ValueError("lane max speeds must be finite and positive")
        setter = getattr(getattr(self._conn, "lane", None), "setMaxSpeed", None)
        if not callable(setter):
            raise SumoSidecarUnavailable("SUMO lane.setMaxSpeed is not available")
        changed: list[str] = []
        try:
            for lane_id, value in requested.items():
                setter(lane_id, value)
                changed.append(lane_id)
        except Exception:
            for lane_id in changed:
                setter(lane_id, before[lane_id])
            raise
        self._snapshot_cache = None
        after = self.edge_lane_max_speeds(edge_id)
        if any(
            not math.isclose(after[lane_id], requested[lane_id])
            for lane_id in requested
        ):
            for lane_id, value in before.items():
                setter(lane_id, value)
            raise RuntimeError(f"SUMO lane speed mutation did not materialize: {edge_id}")
        return {
            "edge_id": str(edge_id),
            "before_lane_max_speeds": before,
            "after_lane_max_speeds": after,
            "sumo_state_mutated": any(
                not math.isclose(before[lane_id], after[lane_id])
                for lane_id in before
            ),
        }

    def lane_disallowed_classes(self, lane_id: str) -> tuple[str, ...]:
        """Read the native vehicle classes currently disallowed on a lane."""
        lane_id = str(lane_id).strip()
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable(
                "lane_disallowed_classes() before start()"
            )
        lane_api = getattr(self._conn, "lane", None)
        getter = getattr(lane_api, "getDisallowed", None)
        if not callable(getter):
            raise SumoSidecarUnavailable(
                "SUMO lane.getDisallowed is not available"
            )
        if Path(self.net_path).is_file():
            self.lane_edge_id(lane_id)
        else:
            ids_getter = getattr(lane_api, "getIDList", None)
            if callable(ids_getter) and lane_id not in {
                str(value) for value in ids_getter()
            }:
                raise ValueError(
                    f"lane_id is not present in the runtime graph: {lane_id}"
                )
        return tuple(sorted({str(value) for value in getter(lane_id)}))

    def lane_allows_vehicle_class(
        self, lane_id: str, vehicle_class: str
    ) -> bool:
        """Return native permission for one class using only target-lane RPCs."""
        lane_id = str(lane_id).strip()
        vehicle_class = str(vehicle_class).strip()
        self.lane_edge_id(lane_id)
        lane_api = getattr(self._conn, "lane", None)
        allowed_getter = getattr(lane_api, "getAllowed", None)
        if not callable(allowed_getter):
            raise SumoSidecarUnavailable("SUMO lane.getAllowed is not available")
        allowed = {str(value) for value in allowed_getter(lane_id)}
        disallowed = set(self.lane_disallowed_classes(lane_id))
        return vehicle_class not in disallowed and (
            not allowed or vehicle_class in allowed
        )

    def route_edges(self, route_id: str) -> tuple[str, ...]:
        """Return the exact edge sequence for one loaded runtime route."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("route_edges() before start()")
        getter = getattr(getattr(self._conn, "route", None), "getEdges", None)
        if not callable(getter):
            raise SumoSidecarUnavailable("SUMO route.getEdges is not available")
        edges = tuple(str(value) for value in getter(str(route_id).strip()))
        if not edges:
            raise ValueError(f"route_id has no runtime edge sequence: {route_id}")
        return edges

    def set_lane_disallowed(
        self,
        *,
        lane_id: str,
        disallowed_classes: tuple[str, ...] | list[str],
    ) -> dict[str, Any]:
        """Apply and verify a native TraCI lane-closure mutation.

        The previous class list is returned so a bounded procedural event can
        restore the exact runtime state.  The sidecar deliberately does not
        accept invented lane ids or silently fall back to an edge-level proxy.
        """
        lane_id = str(lane_id).strip()
        before = self.lane_disallowed_classes(lane_id)
        requested = tuple(sorted({str(value).strip() for value in disallowed_classes if str(value).strip()}))
        lane_api = getattr(self._conn, "lane", None)
        setter = getattr(lane_api, "setDisallowed", None)
        if not callable(setter):
            raise SumoSidecarUnavailable("SUMO lane.setDisallowed is not available")
        setter(lane_id, requested)
        after = self.lane_disallowed_classes(lane_id)
        if after != requested:
            # Restore the exact previous permission state before failing.
            restore = getattr(lane_api, "setDisallowed", None)
            if callable(restore):
                restore(lane_id, before)
            raise RuntimeError(
                f"SUMO lane disallowed state did not materialize: {lane_id}"
            )
        self._snapshot_cache = None
        return {
            "lane_id": lane_id,
            "before_disallowed_classes": list(before),
            "after_disallowed_classes": list(after),
            "sumo_state_mutated": before != after,
        }

    def inject_vehicle_from_route(
        self,
        *,
        vehicle_id: str,
        route_id: str,
    ) -> dict[str, Any]:
        """Inject one deterministic vehicle using a loaded native route.

        This is deliberately a narrow procedural-event primitive, not a general
        simulator control API.  The route must already exist in the runtime
        route inventory, and the caller supplies a deterministic vehicle ID.
        SUMO owns the subsequent physics; the return value records only the
        accepted native mutation and the runtime route identity.
        """
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable(
                "inject_vehicle_from_route() before start()"
            )
        vehicle_id = str(vehicle_id).strip()
        route_id = str(route_id).strip()
        if not vehicle_id:
            raise ValueError("vehicle_id must be non-empty")
        if not route_id:
            raise ValueError("route_id must be non-empty")
        if route_id not in set(self.route_ids()):
            raise ValueError(f"route_id is not present in the runtime graph: {route_id}")
        vehicle_api = getattr(self._conn, "vehicle", None)
        add = getattr(vehicle_api, "add", None)
        if not callable(add):
            raise SumoSidecarUnavailable("SUMO vehicle.add is not available")
        id_getter = getattr(vehicle_api, "getIDList", None)
        if callable(id_getter) and vehicle_id in {str(value) for value in id_getter()}:
            raise ValueError(f"vehicle_id already exists: {vehicle_id}")
        add(
            vehicle_id,
            route_id,
            depart="now",
            departLane="best",
            departPos="base",
            departSpeed="max",
        )
        self._snapshot_cache = None
        return {
            "vehicle_id": vehicle_id,
            "route_id": route_id,
            "depart": "now",
            "sumo_state_mutated": True,
            "vehicle_present_after_add": (
                vehicle_id in {str(value) for value in id_getter()}
                if callable(id_getter)
                else None
            ),
        }

    def inject_demand_surge(
        self,
        *,
        source_vehicle_id: str,
        event_id: str,
        vehicle_count: int,
    ) -> dict[str, Any]:
        """Clone a loaded vehicle route for legacy procedural-event callers.

        New code should prefer :meth:`inject_vehicle_from_route`, which binds
        directly to the runtime route inventory.  This compatibility primitive
        keeps older adapters/test doubles working while still deriving the
        route and vehicle type from live TraCI state; it never accepts a route
        or edge list invented by a scenario declaration.
        """
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("inject_demand_surge() before start()")
        source_vehicle_id = str(source_vehicle_id).strip()
        event_id = str(event_id).strip()
        count = max(1, int(vehicle_count))
        vehicle_api = getattr(self._conn, "vehicle", None)
        route_api = getattr(self._conn, "route", None)
        get_route = getattr(vehicle_api, "getRoute", None)
        get_route_id = getattr(vehicle_api, "getRouteID", None)
        get_type_id = getattr(vehicle_api, "getTypeID", None)
        add_route = getattr(route_api, "add", None)
        add_vehicle = getattr(vehicle_api, "add", None)
        id_getter = getattr(vehicle_api, "getIDList", None)
        if not all(
            callable(value)
            for value in (get_route, get_route_id, get_type_id, add_route, add_vehicle)
        ):
            raise SumoSidecarUnavailable("SUMO route-clone APIs are not available")
        source_route_edges = [
            str(value) for value in get_route(source_vehicle_id) if str(value)
        ]
        source_route_id = str(get_route_id(source_vehicle_id))
        type_id = str(get_type_id(source_vehicle_id))
        if not source_route_edges or not source_route_id or not type_id:
            raise ValueError("source vehicle has no cloneable native route")
        route_suffix = hashlib.sha256(event_id.encode("utf-8")).hexdigest()[:12]
        overlay_route_id = f"dt-procedural-route-{route_suffix}"
        add_route(overlay_route_id, tuple(source_route_edges))
        existing = (
            {str(value) for value in id_getter()}
            if callable(id_getter)
            else set()
        )
        overlay_vehicle_ids: list[str] = []
        for offset in range(count):
            vehicle_id = f"dt-procedural-{route_suffix}-{offset}"
            if vehicle_id in existing:
                raise ValueError(f"overlay vehicle already exists: {vehicle_id}")
            add_vehicle(
                vehicle_id,
                overlay_route_id,
                type_id,
                depart="now",
                departLane="best",
                departPos="base",
                departSpeed="max",
            )
            overlay_vehicle_ids.append(vehicle_id)
            existing.add(vehicle_id)
        before_ids = sorted(
            {str(value) for value in id_getter()} if callable(id_getter) else set()
        )
        self._snapshot_cache = None
        return {
            "sumo_state_mutated": True,
            "injected_vehicle_count": len(overlay_vehicle_ids),
            "overlay_vehicle_ids": overlay_vehicle_ids,
            "source_vehicle_id": source_vehicle_id,
            "source_route_id": source_route_id,
            "source_route_edges": source_route_edges,
            "overlay_route_id": overlay_route_id,
            "before_state": {
                "vehicle_ids": sorted(existing - set(overlay_vehicle_ids)),
            },
            "after_state": {"vehicle_ids": before_ids},
        }

    def vehicle_ids(self) -> tuple[str, ...]:
        """Return currently active runtime vehicle identifiers."""
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("vehicle_ids() before start()")
        getter = getattr(getattr(self._conn, "vehicle", None), "getIDList", None)
        if not callable(getter):
            raise SumoSidecarUnavailable("SUMO vehicle.getIDList is not available")
        return tuple(sorted({str(value) for value in getter() if str(value)}))

    def snapshot(self, tick: int = 0) -> dict[str, Any]:
        """Return a cached state pull; one real query per benchmark tick (§9).

        The cache key is the benchmark ``tick`` so repeated calls within a tick
        return identical bytes (within-tick fog purity). The actual field set is
        intentionally minimal here; ``sumo_backend.py`` enriches it.
        """
        if not self._started or self._conn is None:
            raise SumoSidecarUnavailable("snapshot() before start()")
        if self._snapshot_cache is not None and self._snapshot_tick == tick:
            return self._snapshot_cache
        conn = self._conn
        snap: dict[str, Any] = {
            "tick": int(tick),
            "transport": self._transport,
            "n_vehicles": int(conn.vehicle.getIDCount()),
            "sim_time": float(conn.simulation.getTime()),
            "arrived": int(conn.simulation.getArrivedNumber()),
            "departed": int(conn.simulation.getDepartedNumber()),
        }
        try:
            snap["network_counts"] = self.network_counts()
        except SumoSidecarUnavailable:
            snap["network_counts"] = {"n_lanes": None, "n_edges": None}
        self._snapshot_cache = snap
        self._snapshot_tick = tick
        return snap

    def close(self) -> None:
        """Tear down the connection and force-kill any orphan process.

        ``finally``-guarded in every branch so a transport error during
        teardown can never leak a TraCI/Docker child (draft risk #7).
        """
        try:
            if self._conn is not None:
                with suppress(Exception):
                    self._conn.close()
        finally:
            self._conn = None
            proc = self._proc
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force-kill orphan (Docker run / sumo binary).
                        try:
                            proc.kill()
                        except ProcessLookupError:
                            pass
                        else:
                            with suppress(
                                ProcessLookupError,
                                PermissionError,
                            ):
                                os.kill(proc.pid, signal.SIGKILL)
                            with suppress(subprocess.TimeoutExpired):
                                proc.wait(timeout=1)
                except Exception:
                    pass
            self._proc = None
            self._traci_label = None
            self._started = False
            self._snapshot_cache = None
            self._snapshot_tick = -1

    # Context-manager sugar so callers get orphan-safe close for free.
    def __enter__(self) -> SumoSidecar:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

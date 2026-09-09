"""Portable scorer inputs and a separate, hash-bound offline rescore output.

No backend or provider is invoked. Completed-runtime snapshots are diagnostic
inputs for counterfactual repair; only scoring-input snapshots are sufficient
for this scorer-only command. Neither replaces historical episode outcomes.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from core import EvidenceLogger
from core.ethical_dilemma import (
    Dilemma, EthicalDilemmaManager, EthicalEpisodeRecord, MoralChoice, MoralOption,
)
from core.stakeholder_trust import StakeholderGroup, StakeholderTrustManager
from evaluation.scorer import ScoringInputs, score_episode


def encode_json(value: Any) -> Any:
    """Preserve invalid native numeric values for diagnosis in strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"__nonfinite_float__": str(value)}
    if isinstance(value, dict):
        return {str(key): encode_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [encode_json(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"unsupported scoring snapshot value: {type(value).__name__}")


def decode_json(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__nonfinite_float__"}:
            if value["__nonfinite_float__"] not in {"nan", "inf", "-inf"}:
                raise ValueError("invalid nonfinite numeric marker")
            return float(value["__nonfinite_float__"])
        return {key: decode_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_json(item) for item in value]
    return value


def snapshot_inputs(inputs: ScoringInputs) -> dict[str, Any]:
    excluded = {"evidence_logger", "stakeholder_mgr", "dilemma_mgr"}
    payload = {field.name: getattr(inputs, field.name) for field in fields(inputs)
               if field.name not in excluded}
    evidence = inputs.evidence_logger
    payload["evidence_logger"] = None if evidence is None else {
        "episode_id": evidence.episode_id, "items": evidence.to_jsonable()}
    trust = inputs.stakeholder_mgr
    # Both stakeholder dimensions consume snapshot() readings only; preserve
    # precisely those rounded terminal readings, without serializing methods.
    payload["stakeholder_mgr"] = None if trust is None else {
        key: asdict(reading) for key, reading in trust.snapshot().items()}
    payload["dilemma_mgr"] = None if inputs.dilemma_mgr is None else asdict(inputs.dilemma_mgr.record)
    return encode_json(payload)


def restore_inputs(payload: dict[str, Any]) -> ScoringInputs:
    values = decode_json(payload)
    if set(values) != {field.name for field in fields(ScoringInputs)}:
        raise ValueError("scoring snapshot fields do not match the scorer contract")
    raw_evidence = values.pop("evidence_logger")
    evidence = None
    if raw_evidence is not None:
        evidence = EvidenceLogger(raw_evidence["episode_id"])
        for item in raw_evidence["items"]:
            identifier = evidence.log(kind=item["kind"], tick=item["tick"],
                                      payload=item["payload"], source=item["source"])
            if identifier != item["evidence_id"]:
                raise ValueError("scoring snapshot evidence identity mismatch")
    raw_trust = values.pop("stakeholder_mgr")
    trust = None
    if raw_trust is not None:
        trust = StakeholderTrustManager()
        for key, reading in raw_trust.items():
            if key != reading["group_id"]:
                raise ValueError("stakeholder identity mismatch")
            trust.register(StakeholderGroup(key, key, baseline_trust=reading["trust"]))
            if reading["last_event"] is not None:
                trust._last_event[key] = (reading["last_event"], reading["last_event_tick"])
    raw_dilemmas = values.pop("dilemma_mgr")
    dilemmas = None
    if raw_dilemmas is not None:
        dilemmas = EthicalDilemmaManager()
        triggered = []
        for item in raw_dilemmas["dilemmas_triggered"]:
            definition = dict(item)
            definition["options"] = [MoralOption(**option) for option in definition["options"]]
            triggered.append(Dilemma(**definition))
        dilemmas._record = EthicalEpisodeRecord(
            dilemmas_triggered=triggered,
            choices={key: MoralChoice(**choice) for key, choice in raw_dilemmas["choices"].items()},
            consequence_realized=raw_dilemmas["consequence_realized"],
        )
    return ScoringInputs(**values, evidence_logger=evidence, stakeholder_mgr=trust, dilemma_mgr=dilemmas)


def write_rescore(source: Path, expected_sha256: str, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError("scoring snapshot hash mismatch")
    envelope = json.loads(raw)
    if (envelope.get("schema_version") != "episode_scoring_snapshot_v1"
            or envelope.get("kind") != "scoring_inputs"):
        raise ValueError("a complete scoring-input snapshot is required")
    payload = envelope["payload"]
    inputs = restore_inputs(payload["inputs"])
    if payload["identity"]["scenario_signature"] != inputs.scenario_signature:
        raise ValueError("scoring snapshot scenario identity mismatch")
    from core.implementation_identity import implementation_identity

    result = {"schema_version": "offline_episode_rescore_v1",
              "source_snapshot_sha256": digest, "source_identity": payload["identity"],
              "scoring_implementation": implementation_identity(Path(__file__).resolve().parents[1]),
              "score": score_episode(inputs).to_dict(), "formal_completion_claimed": False}
    # Exclusive creation prevents replacing either a historical result or a
    # previous repair. The output is not merged into episodes.jsonl.
    with output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_rescore(args.input, args.sha256, args.output)


if __name__ == "__main__":
    main()

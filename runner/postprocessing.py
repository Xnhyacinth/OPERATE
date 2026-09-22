"""Provider-free recovery of completed episodes into separate diagnostic outputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from core import Action, ToolCall, ToolRegistry, ToolSpec
from core.implementation_identity import implementation_identity
from domains.registry import get_domain_spec
from evaluation.scoring_snapshot import decode_json, encode_json, restore_inputs


def _unavailable_handler(*args, **kwargs):
    raise RuntimeError("recovered tool metadata cannot execute controls")


def _read_binding(binding):
    if binding is None:
        return None
    raw = Path(binding['path']).read_bytes()
    if (hashlib.sha256(raw).hexdigest() != binding['sha256']
            or ('byte_count' in binding and len(raw) != binding['byte_count'])):
        raise ValueError('session artifact hash mismatch')
    return raw


def _resolve_binding(binding, source):
    if binding is None:
        return None
    # Retry quarantine moves a complete directory without rewriting historical
    # descriptors. Only use a moved sibling when its original hash still matches.
    for path in dict.fromkeys((Path(binding['path']), source.parent / Path(binding['path']).name)):
        candidate = {**binding, 'path': str(path)}
        try:
            _read_binding(candidate)
        except (FileNotFoundError, ValueError):
            continue
        return candidate
    raise ValueError('session artifact hash mismatch or unavailable')


def recover_completed_episode(source: Path, expected_sha256: str, *,
                              expected_identity: dict[str, Any], output: Path) -> dict[str, Any]:
    """Recompute all episode scoring without invoking an agent or overwriting history.

    ``expected_identity`` must be the complete original snapshot identity pinned
    by the caller, not the current implementation identity. Repairs always carry
    both identities and never claim formal completion or merge into a batch.
    """
    from runner import episode

    if output.exists():
        raise FileExistsError(output)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError('completed runtime hash mismatch')
    envelope = json.loads(raw)
    if (envelope.get('schema_version') != 'episode_scoring_snapshot_v1'
            or envelope.get('kind') != 'completed_runtime'):
        raise ValueError('a completed runtime snapshot is required')
    payload = decode_json(envelope['payload'])
    identity = payload['identity']
    if identity != expected_identity:
        raise ValueError('completed runtime identity mismatch')
    context = payload.get('postprocessing_context')
    legacy = context is None
    if legacy:
        context = _legacy_citylearn_context(payload)
    if context.get('schema_version') != 'completed_postprocessing_context_v1':
        raise ValueError('completed postprocessing context unavailable')
    scenario = payload['scenario']
    spec = get_domain_spec(scenario.get('domain'))
    seed = identity['seed']
    if episode.recompute_signature_with_seed(scenario, seed, spec) != identity['scenario_signature']:
        raise ValueError('completed runtime scenario identity mismatch')
    for key in ('semantic_ledger_artifact', 'provider_audit_artifact'):
        payload[key] = _resolve_binding(payload.get(key), source)
    inputs = restore_inputs(payload['manager_inputs'])
    registry = None
    if context['tool_registry'] is not None:
        registry = ToolRegistry()
        for definition in context['tool_registry']:
            registry.register(ToolSpec(**definition, handler=_unavailable_handler))
    env = SimpleNamespace(
        evidence=inputs.evidence_logger, stakeholders=inputs.stakeholder_mgr,
        dilemmas=inputs.dilemma_mgr, tick=context['tick'], _tools=registry,
        get_tool_specs=lambda: context['tool_specs'],
        readonly_tool_names=lambda: context['readonly_tool_names'],
    )
    agent = SimpleNamespace(get_interaction_stats=lambda: context['llm_stats'],
                            progress=lambda: context['checkpoint_progress'])
    actions = [Action(tool_calls=[ToolCall(**call) for call in item['actions']],
                      dominant=item['dominant_action'], assistant_text=item['assistant_text'],
                      rationale=item['rationale']) for item in payload['actions']]
    loop_result = {**context['loop_result'], 'actions': actions}
    settings = payload['counterfactual_settings']
    recompute_identity = implementation_identity(episode.REPO_ROOT)
    logger = _recovery_logger(source, payload, output)
    source_session_artifacts = {key: payload.get(key) for key in (
        'semantic_ledger_artifact', 'provider_audit_artifact')}
    for key, suffix in (('semantic_ledger_artifact', 'semantic_ledger'),
                        ('provider_audit_artifact', 'provider_audit')):
        binding = payload.get(key)
        if binding is not None:
            destination = logger.output_dir / f'{logger.episode_id}.{suffix}.jsonl'
            with destination.open('xb') as handle:
                handle.write(_read_binding(binding))
            payload[key] = {**binding, 'path': str(destination)}
    result = episode._postprocess_completed_episode(
        scenario=scenario, agent_name=identity['agent_name'], env=env, spec=spec,
        seed=seed, agent_kwargs=identity['agent_config'], trajectory_dir=logger.output_dir,
        counterfactual_masking=settings['masking_policy'],
        per_action_attribution=settings['per_action'], per_action_cap=settings['per_action_cap'],
        per_action_group_attribution=settings['per_action_groups'],
        per_action_group_cap=settings['per_action_group_cap'],
        within_tick_interaction=context['within_tick_interaction'],
        checkpoint_path=source if context['checkpoint_progress'] is not None else None,
        agent=agent, agent_extras=context['agent_extras'], logger=logger,
        loop_result=loop_result, gt=payload['ground_truth'], foresight=payload['foresight'],
        backend_records=payload['backend_tick_records'], realized=payload['realized_events'],
        snapshot_identity={**identity, "implementation": recompute_identity},
        recovery_metadata={"source_identity": identity, "recompute_identity": recompute_identity,
                           "formal_completion_claimed": False},
        unavailable_trajectory_fields=_LEGACY_UNAVAILABLE if legacy else (),
        completed_runtime_artifact={'path': str(source), 'sha256': digest,
                                    'schema_version': envelope['schema_version'], 'byte_count': len(raw)},
        lp_optimum=context['lp_optimum'],
        optimality_objective_component=context['optimality_objective_component'],
        session_artifacts=(payload['structured_memory'], payload['semantic_ledger_artifact'],
                           payload['provider_audit_artifact']),
    )
    if legacy:
        result['evaluation_protocol']['within_tick_interaction'] = None
    ending_identity = implementation_identity(episode.REPO_ROOT)
    runtime_unchanged = (recompute_identity['evaluation_runtime_sha256']
                         == ending_identity['evaluation_runtime_sha256'])
    repaired = {
        'schema_version': 'offline_completed_episode_recovery_v1',
        'source_snapshot_sha256': digest, 'source_identity': identity,
        'source_session_artifacts': source_session_artifacts,
        'recompute_identity': recompute_identity,
        'ending_implementation_identity': ending_identity,
        'runtime_implementation_unchanged': runtime_unchanged,
        'formal_completion_claimed': False,
        'legacy_context_reconstructed': legacy,
        'same_contract_recovery': (not legacy and runtime_unchanged
            and payload.get('trajectory_header') is not None and
            identity['implementation']['evaluation_runtime_sha256'] ==
            recompute_identity['evaluation_runtime_sha256']),
        'result': result,
        'evidence': env.evidence.to_jsonable() if env.evidence is not None else None,
    }
    with output.open('x') as handle:
        json.dump(encode_json(repaired), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    return repaired


def _recovery_logger(source, payload, output):
    """Copy immutable trajectory bytes; write new scoring artifacts separately."""
    from data import EpisodeHeader, TrajectoryLogger

    binding = payload.get('trajectory_artifact')
    prefix = str(source).removesuffix('.completed_runtime.json')
    summary_path = Path(prefix + '.summary.json')
    if binding is None:
        if not summary_path.is_file():
            raise ValueError('legacy source trajectory binding unavailable')
        binding = json.loads(summary_path.read_text())['trajectory_summary']['trajectory_artifact']
    binding = _resolve_binding(binding, source)
    trace = _read_binding(binding)
    target = output.with_suffix('.artifacts')
    target.mkdir(exist_ok=False)
    logger = TrajectoryLogger(episode_id='recovered')
    for line in trace.splitlines():
        entry = json.loads(line)
        logger.log_step(**{key: value for key, value in entry.items() if key != 'timestamp_utc'})
        logger.entries.clear()
    logger.output_dir = target
    logger._claim_output()
    (target / 'recovered.trajectory.jsonl').write_bytes(trace)
    header_path = Path(prefix + '.header.json')
    if payload.get('trajectory_header') is not None:
        header = payload['trajectory_header']
    elif header_path.is_file():
        header = json.loads(header_path.read_text())
    else:
        header = None
    if header is not None:
        if (header['scenario_signature'] != payload['identity']['scenario_signature']
                or header['seed'] != payload['identity']['seed']):
            raise ValueError('source trajectory header identity mismatch')
        logger.set_header(EpisodeHeader(**header))
    return logger


_LEGACY_UNAVAILABLE = (
    'multi_turn', 'within_tick_interaction', 'event_adaptive_autonomy',
    'terminal_integrity', 'event_contract', 'transition_ingestion',
    'initiative_lead_ticks', 'plan_review_honored_rate', 'llm',
)


def _legacy_citylearn_context(payload):
    """Recover scoring-sufficient old CityLearn snapshots, label missing telemetry.

    CityLearn reset never computes/mutates a cached oracle; its seed backend
    config is the oracle input. The terminal tick is explicitly in ground truth.
    Missing runtime diagnostics cannot be reconstructed and remain unavailable.
    """
    from domains.registry import reference_optimum_from_backend_config, reference_optimum_objective_component

    scenario = payload.get('scenario') or {}
    if scenario.get('domain') != 'building_energy' or scenario.get('backend_kind') != 'citylearn':
        raise ValueError('completed postprocessing context unavailable for legacy snapshot')
    gt = payload['ground_truth']
    if 'tick' not in gt or gt['tick'] != len(payload['actions']):
        raise ValueError('legacy completed context terminal tick mismatch')
    steps = payload['analysis_steps']
    investigations = [step['info']['extra']['within_tick_investigation'] for step in steps
                      if (step.get('info') or {}).get('extra', {}).get('within_tick_investigation')]
    tool_results = [result for step in [*steps, *investigations] for result in step.get('tool_results', [])]
    loop = {key: {} for key in _LEGACY_UNAVAILABLE}
    loop.update(analysis_steps=steps, stale_observation_records=payload['stale_observation_records'],
                tool_results_ok=sum(result['ok'] is True for result in tool_results),
                tool_results_failed=sum(result['ok'] is False for result in tool_results),
                within_tick_records=investigations, multi_turn_records=[])
    env = SimpleNamespace(seed_obj=SimpleNamespace(backend_config=scenario.get('backend_config') or {}))
    from domains.building_energy.tools import register_building_energy_tools

    registry = ToolRegistry()
    register_building_energy_tools(registry, SimpleNamespace(buildings=list(gt['buildings'])), env)
    definitions = [{key: getattr(registry.get(name), key) for key in (
        'name', 'description', 'parameters', 'state_changing', 'semantic_role',
        'native_target_kind', 'actuator_family', 'cost_units')} for name in registry.names()]
    return {
        'schema_version': 'completed_postprocessing_context_v1', 'loop_result': loop,
        'tick': gt['tick'], 'agent_extras': None, 'llm_stats': None,
        'checkpoint_progress': None, 'within_tick_interaction': False,
        'lp_optimum': reference_optimum_from_backend_config(env),
        'optimality_objective_component': reference_optimum_objective_component(env, default='energy_cost'),
        'tool_registry': definitions, 'tool_specs': registry.openai_schemas(),
        'readonly_tool_names': [name for name in registry.names() if not registry.get(name).state_changing],
    }

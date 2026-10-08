"""Predeclared deployment-system comparisons; never equal-compute claims.

The launcher freezes this descriptor before execution. This is an auditable
campaign contract, not a signature against adversarial file fabrication.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

KIND = 'predeclared_route_native_budget_v1'
SCHEDULING = ('implementation_policy', 'seed_mode', 'scheduler_mode',
              'requested_concurrency', 'effective_concurrency',
              'max_workers_requested', 'max_workers_effective')
CONCURRENCY = SCHEDULING[3:]
SHARED = ('implementation_tree_sha256', 'suite_manifest_sha256', 'interaction_mode', 'scoring_version') + SCHEDULING
ROUTE_FIELDS = frozenset((
    'model', 'provider', 'base_url', 'api_version', 'api_version_env',
    'responses_base_url', 'responses_base_url_env', 'private_provider_route_sha256',
    'api_mode', 'stream_chat_completions', 'max_tokens',
    'model_context_window_tokens', 'model_max_output_tokens', 'reasoning_effort',
    'reasoning_effort_format', 'thinking_type', 'tool_choice_supported',
    'accepted_response_models', 'extra_header_names',
))


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def _agent_config(profile):
    from baselines import LLMConfig
    from runner.episode import _public_agent_config
    values = {k: v for k, v in profile.items() if k in LLMConfig.__dataclass_fields__}
    for key in ('provider_rpm_limit', 'provider_rpd_limit'):
        values[key] = values.get(key) or 0
    values['extra_headers'] = {k: '[redacted]' for k in profile.get('extra_header_names', [])}
    return _public_agent_config({'config': LLMConfig(**values)})


def build_protocol(run_configs, *, suite_sha256, pass_id='pass-0'):
    """Compile exact CLI-resolved profiles, never infer a profile hash from a label."""
    if not run_configs or not suite_sha256 or not pass_id:
        raise ValueError('comparison protocol requires suite, models and pass')
    declarations = {}
    shared = shared_profile = None
    for model, cfg in run_configs.items():
        conditions = {key: cfg.get(key) for key in SHARED}
        if (any(not isinstance(v, str) or not v for k, v in conditions.items() if k not in CONCURRENCY)
                or any(type(conditions[k]) is not int or conditions[k] < 1 for k in CONCURRENCY)):
            raise ValueError('comparison protocol shared identity missing')
        if (conditions['requested_concurrency'] != conditions['max_workers_requested']
                or conditions['effective_concurrency'] != conditions['max_workers_effective']):
            raise ValueError('comparison protocol shared concurrency aliases mismatch')
        profile = cfg['agent_profile_identity_by_model'][model]
        profile_hash = cfg['agent_profile_sha256_by_model'][model]
        if profile.get('model') != model or _hash(profile) != profile_hash:
            raise ValueError('comparison protocol profile hash mismatch')
        common = {k: v for k, v in profile.items() if k not in ROUTE_FIELDS}
        if shared is not None and (conditions != shared or common != shared_profile):
            raise ValueError('comparison protocol shared conditions mismatch')
        shared, shared_profile = conditions, common
        treatment = cfg['agent_treatment_sha256_by_model'][model]
        if not treatment or not cfg.get('run_semantics_fingerprint'):
            raise ValueError('comparison protocol treatment missing')
        declarations[model] = dict(
            profile=profile,
            agent_config=_agent_config(profile),
            row_identity=dict(agent_profile_sha256=profile_hash,
                              agent_treatment_sha256=treatment,
                              run_semantics_fingerprint=cfg['run_semantics_fingerprint'] + f':agent-{treatment}'),
        )
    return dict(kind=KIND, suite_sha256=suite_sha256, pass_id=pass_id,
                shared_conditions=shared, shared_profile=shared_profile,
                models=declarations,
                interpretation='Deployment systems at predeclared route-native budgets; not equal compute.')


def load_protocol(config, *, suite_sha256, models):
    descriptor = config.get('comparison_protocol')
    if descriptor is None:
        return None, None
    if config.get('comparison_policy', 'strict') != 'strict':
        raise ValueError('comparison protocol requires strict policy')
    raw = Path(descriptor['path']).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if not descriptor.get('sha256') or digest != descriptor['sha256']:
        raise ValueError('comparison protocol hash mismatch')
    protocol = json.loads(raw)
    if protocol.get('kind') != KIND or protocol.get('suite_sha256') != suite_sha256:
        raise ValueError('comparison protocol kind or suite mismatch')
    if set(protocol.get('models', {})) != set(models):
        raise ValueError('comparison protocol model population mismatch')
    # A protocol campaign scores original batch files, not copied rows with an
    # arbitrary self-reported source path. The actual input selects its sidecar.
    for paths in config.get('runs', {}).values():
        for path in paths:
            source = Path(path).resolve()
            for line in source.read_bytes().splitlines():
                if line.strip():
                    item = json.loads(line)
                    if Path(item.get('_offline_source_path', str(source))).resolve() != source:
                        raise ValueError('comparison protocol requires original batch input paths')
    # Recompile declarations to reject incomplete or internally inconsistent files.
    configs = {}
    for model, declaration in protocol['models'].items():
        ident = declaration['row_identity']
        suffix = ':agent-' + ident['agent_treatment_sha256']
        semantics = ident['run_semantics_fingerprint']
        if not semantics.endswith(suffix):
            raise ValueError('comparison protocol treatment suffix mismatch')
        configs[model] = dict(protocol['shared_conditions'],
            agent_profile_identity_by_model={model: declaration['profile']},
            agent_profile_sha256_by_model={model: ident['agent_profile_sha256']},
            agent_treatment_sha256_by_model={model: ident['agent_treatment_sha256']},
            run_semantics_fingerprint=semantics[:-len(suffix)])
    rebuilt = build_protocol(configs, suite_sha256=suite_sha256, pass_id=protocol['pass_id'])
    if rebuilt != protocol:
        raise ValueError('comparison protocol declaration mismatch')
    return protocol, digest


def validate_row(protocol, model, row):
    if model not in protocol['models'] or row.get('model') != model:
        raise ValueError('comparison protocol undeclared model')
    expected = dict(protocol['shared_conditions'], **protocol['models'][model]['row_identity'],
                    pass_id=protocol['pass_id'])
    scorer = expected.pop('scoring_version')
    if (row.get('score') or {}).get('scoring_version') != scorer:
        raise ValueError('comparison protocol scoring version mismatch')
    sidecar = None
    source = row.get('_offline_source_path')
    if source:
        path = Path(source).resolve().parent / 'run_config.json'
        if path.is_file():
            sidecar = json.loads(path.read_bytes())
    for field, value in expected.items():
        actual_value = row.get(field)
        if field in SCHEDULING:
            if sidecar is not None and sidecar.get(field) != value:
                raise ValueError(f'comparison protocol run_config {field} mismatch')
            if field not in row:
                if sidecar is None:
                    raise ValueError('comparison protocol missing original run_config scheduling evidence')
                actual_value = sidecar.get(field)
        if actual_value != value:
            raise ValueError(f'comparison protocol {field} mismatch')
    tree = expected['implementation_tree_sha256']
    if any(row.get(k) != tree for k in ('implementation_tree_sha256_start', 'implementation_tree_sha256_end')):
        raise ValueError('comparison protocol runtime mismatch')
    actual = row.get('agent_config')
    if isinstance(actual, dict) and isinstance(actual.get('config'), dict):
        actual = {**actual, 'config': {**actual['config']}}
        # Credential environment variable names are not request semantics.
        actual['config']['api_key_env'] = 'OPENAI_API_KEY'
    if actual != protocol['models'][model]['agent_config']:
        raise ValueError('comparison protocol agent_config mismatch')

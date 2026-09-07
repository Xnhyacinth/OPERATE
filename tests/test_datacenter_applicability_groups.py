from domains.datacenter.source_native_candidates import stakeholder_equity_applicability


def test_singleton_source_users_do_not_claim_cross_tenant_equity():
    body = {'backend_kind': 'alibaba_trace_sim', 'backend_config': {'jobs': [
        {'user': 'u1'}, {'user': 'u1'}]}}
    assert stakeholder_equity_applicability(body)['applicable'] is False
    body['backend_config']['jobs'].append({'user': 'u2'})
    assert stakeholder_equity_applicability(body)['applicable'] is True

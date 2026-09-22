from evaluation.reference_compatibility import provider_only_change


def test_only_named_provider_methods_and_literal_metadata_are_compatible():
    source = 'STATIC_MODEL_TOOL_CHOICE_SUPPORT: dict = {"a": True}\nclass LLMAgent:\n def _call_openai_protocol_repair(self):\n  return 0\n def act(self):\n  return 1\n'
    assert provider_only_change(
        source,
        source.replace("return 0", "return 2").replace('"a": True', '"a": False'),
    )
    assert not provider_only_change(source, source.replace("return 1", "return 2"))
    assert not provider_only_change(source, source.replace('{"a": True}', "evil()"))
    assert not provider_only_change(source, source + "\ninstall_global_hook()\n")

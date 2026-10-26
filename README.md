# Anonymous source review

This is a strict offline source-inspection snapshot.

It contains selected computational source for event handling, evidence,
execution control, reference policies and scoring. Retained implementation files
are unchanged. Files carrying network addresses or identifying attribution are
excluded, rather than edited into different scientific inputs.

This is not an import-complete or executable benchmark distribution. Some source
imports refer to omitted integration modules. The corpus, frozen scoring inputs,
native and model transports, dependency environment and original external notices
must be supplied as separate local materials. The full installation and benchmark
commands do not apply to this snapshot.

## Offline checks

The review tools use only the standard library. They do not install the retained
computational modules or fetch dependencies.

```bash
uv sync --offline --frozen --python 3.13
uv run --offline --frozen python scripts/check_snapshot.py
uv run --offline --frozen python -m unittest discover -s tests -v
```

The checker validates the file inventory, parses every included source file and
rejects network addresses, contact addresses, unsafe paths and distribution
symlinks. The small regression suite checks selected retained primitives. Passing
these checks does not reproduce the external benchmark or certify model results.

Runtime output and caches are local and are excluded from the source inventory.
The generic license notice applies to the retained original implementation.

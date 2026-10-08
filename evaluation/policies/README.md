# Immutable offline scoring inputs

The 0.27/0.28 policies retain their original hashes. Their 141 Lite
scenario YAMLs and suite remain at the exact versioned paths declared
by the policy, separately from the flattened runtime catalog under
benchmark/. These are current policy inputs, not historical runs.

offline_source_inventory.json records copied inputs and required upstream
source files under works/. Restore those external files with the runtime
companion/setup before offline scoring; hashes must match the inventory.
Use scripts/setup_eval_env.sh from the public checkout: CityLearn and
pglib-uc are fetched at the runtime manifest pinned upstream commits;
DynaSched files are restored from its hash-bound source asset archive.
Original trajectory journals and provider evidence are supplied separately
by the evaluator. No raw trajectories are included in this snapshot.

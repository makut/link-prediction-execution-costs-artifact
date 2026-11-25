Explicit independent PENCIL V3 REPLAY timer correction, addendum V2

This is a NEW derived audit, not an edit to the frozen110-check verifier or
scientific producer. The first actual frozen audit ran once, exited2 with
"REPLAY must explicitly bypass sampler", and produced no numerical report.
CORRECTION.json binds that original attempt and stderr by whole bytes/SHA.

The failed predicate incorrectly required sampler_seconds.max_s==0 in REPLAY.
Frozen SamplerHook bypasses the real sampler for frozenS but measures np.array
plus torch.from_numpy materialization. QueryRuntime.raw and PipelineDataset
propagate this duration as sampler_seconds. Both producer files and the exact
four-function AST callchain are pinned. The one-shot actual synthetic callspy
uses the real frozen SamplerHook, a real sampler that raises if called in
REPLAY, and an online counter. It proves0 REPLAY calls,1 ONLINE call, exact
frozen int64 IDs, and a valid positive materialization timer on a tiny fixture.
It is local CPU verification, not a new graph, native sparse, CUDA or timing
experiment. Do not execute callspy_once.py again: its receipt is exclusive.

audit_addendum_v2.py verifies every original frozen core source before import.
It derives only benchmark_receipt by replacing exactly one source statement:
the invalid zero-time predicate becomes a check of the declared REPLAY route
and its materialization summary. Every other function-body byte is retained.
All original stage count/finite/unit/quantile checks, archive/source/launch/
recovery/environment/policy/trace/CSR/value/model/full-score/QPS/median/guard
checks remain unchanged. Output adds the explicit correction and limitations.
Neither actual producer evidence nor actual sweep times are changed.

After independent peer review, the authorized actual invocation is:
python3 -B work/vldb_pencil_replay_addendum_v2/audit_addendum_v2.py \
  --export outputs/seal_after_travel_systems_final_20261125/pencil_bridge_v3_final_readonly_v1 \
  --output outputs/seal_after_travel_systems_final_20261125/pencil_bridge_v3_independent_audit_addendum_v2_actual_v1 \
  --root-launch work/vldb_gpu_recovery/pencil_queue_v3_launch.json

This new actual audit is also a one-shot run into fresh output. The original
failed receipt and original verifier stay intact. Any new failure stops
performance interpretation; there is no catch-and-ignore or producer repair.

Regression suite: original110 tests plus23 addendum tests. New synthetic
coverage includes nonzero REPLAY materialization in a complete48-job export,
all unchanged full-score/configuration gates, nine false-success corruptions,
callspy false claims, original failed-evidence preservation and source output
refusal. Synthetic values are never scientific measurements. Independent peer
should also install the derived benchmark function into the original test
module and rerun its110 tests, exercising the same retained gates directly.

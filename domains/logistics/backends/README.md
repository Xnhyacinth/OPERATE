# Current routing execution contract

`route_sim`, `pyvrp_cvrp`, `pyvrp_vrptw`, and `pyvrp_lastmile` execute
`route_service_quota_v2`: each active vehicle can serve a bounded number of
ordered stops per dispatch wave, subject to actual remaining capacity. Source
coordinates determine route-distance cost; source demand and vehicle capacity
constrain deliveries. The backend identifier `pyvrp_vrptw` is historical.

**Native travel duration, source service duration, and source time-window
feasibility are not implemented.** Snapshots and source traces expose these
capabilities as `applicable: false`. Procedural `due_tick` values are dispatch
wave deadlines, not source time windows. `query_eta` reveals vehicle status and
quota but returns no invented travel ETA. These outputs cannot support a claim
of evaluating full VRPTW timing ability. A future timed implementation requires
an explicit source-time/dispatch-clock mapping and new validation.

Spot procurement creates an actual, empty-route carrier at the depot; its
arrival event gives the ID used by the normal assignment tools. The request's
region is a label, not a simulated geographical service boundary. Carrier
capacity is consumed by delivery, and reserve statistics report remaining
usable standby capacity. Finite breakdown/blockage intervals expire after the
last overlapping interval; no replenishment or physical vehicle repair model
is implied beyond the declared perturbation interval.

Canceling a stop removes it from routes and adds the existing cancellation
charge. It does not erase the unfulfilled demand or end the episode as a
successful delivery. Unmet demand continues to accrue under the same deadline
rule through the remaining execution horizon. Terminal snapshots expose all
unfulfilled and canceled units.

These changes alter state, observations, costs, tools, and stopping behavior.
Old model trajectories are historical evidence, not valid new-run model
results. Fixed-action replay is a diagnostic only; candidate model evaluations
and no-action controls must use the same new implementation. Frozen scenarios
and manifests are intentionally unchanged and do not qualify this new contract
for release.

All routing policies execute exactly the scenario's declared horizon. Serving
the current orders does not end the episode before later arrivals; a temporary
fleet outage does not irreversibly terminate it. Catastrophic fleet failure
means no active usable vehicle and unfulfilled orders **at the horizon
boundary**. Both transient failure and permanent failure continue obligation
accounting until that boundary. Events or procurements scheduled beyond the
horizon cannot extend it. This equal-window contract also applies to no-action
controls; it does not claim long-run performance beyond the declared horizon.

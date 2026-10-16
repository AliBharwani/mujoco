# Feather integration API

`mjfeather.h` exposes the fork's integration API. Since integration11, `mjModel`
carries one fork array, `hfield_collider` (each heightfield's convex collider, see
[smooth heightfield contacts](FEATHER_HFIELD.md)); integration3's three hfield-cache
arrays are gone. Regenerate bindings and use a complete matching bundle. `mjData`
and `mjContact` remain unchanged. The [midphase](FEATHER_MIDPHASE.md) changes
(node masks, class-grouped body trees, covered body pairs, one-pair collisions)
are internal and bit-exact.

- `mjf_enableTimers(1)` installs a native monotonic wall clock in milliseconds
  (`CLOCK_MONOTONIC` on Linux, `QueryPerformanceCounter` on Windows). It is opt-in
  and process-global: it replaces any Python `set_mjcb_time` callback; disabling
  sets NULL, not the previous callback. Enable/disable only when all simulations
  and workers are idle. It does not read, clear or otherwise edit any `mjData`.
  Upstream call counts advance even when the callback is NULL; only durations
  stop accumulating. Consumers must clear both fields at measurement boundaries.
  Timers overlap: STEP contains FORWARD/ADVANCE, POSITION contains its sub-stages,
  and COLLISION contains BROAD/NARROW. Inverse is inactive in ordinary forward
  stepping. Omitted work/timer overhead means an arbitrary sub-stage sum need not
  equal STEP. Wall time includes descheduling; measure overhead before leaving it
  on in production collection. Application call sites own enabling the clock.
- `mjf_buildId` identifies native source + declared dependency pins (including dirty
  source files). `mjf_layout` checks struct sizes and selected field offsets before
  a consumer allocates or dereferences them. Binary hashes/toolchain identity are
  separate provenance: the source ID is common to Linux and Windows.
- Filtered ray and overlap calls use `(contype | conaffinity) & mask`. Disabled
  collision geoms are excluded. Rays retain upstream visibility/static handling.
  Overlap is conservative bounding spheres plus solid plane halfspaces, returning
  the total count even when its caller buffer is too small. It is still O(ngeom).
  Original query exports and upstream `mj_ray` semantics are preserved.
- Heightfield writes validate the entire finite input and exact count, clamp to
  [0,1], and wake overlapping sleeping islands on changes (since integration4;
  see [regional wakes](FEATHER_REGION_WAKE.md)). Identical writes do not wake. For
  shared models, the caller must wake every other live `mjData` and synchronize
  writes against stepping. Other runtime model edits must explicitly wake too.
- Guarded substeps preserve Feather's post-step joint/free-root velocity clamps,
  consume divergence warnings each step and apply a one-shot force only once.
  Auto-reset already clears forces; do not leave a negative impulse afterward.
  Bulk velocity readback uses `mj_objectVelocity` and preserves BODY vs XBODY.
  Callers own disjoint data during stepping/readback; no internal worker pools.

The 3.12 port uses `mj_wakeIsland` (with a model-edit log reason) instead of 3.9's
`mj_wakeTree`. Both call the upstream sleep-array maintenance helper after waking.
No new solver algorithm, fast-math, or global sleep setting is introduced.

Feather has matching ctypes integration tests and Unity has pre-load ABI checks
and core tests. Build recipes and package manifests live in Feather's
`tools/mujoco/` and `dist/mujoco/`. Keep both native platforms and the wheel built
from one source; a GCC-built core cannot safely be swapped into an official Linux
Python wheel whose C++ bindings use libc++.

# Feather: what this fork adds to MuJoCo 3.12.0

This fork is the physics core of Feather's Unity game. It adds functions under the `mjf_` prefix,
a smooth convex × heightfield collider, a faster broadphase and midphase, positive-margin convex
manifolds and opt-in contact reuse; everything else is upstream 3.12.0. The one layout change is a
trailing pointer in `mjData`, `contactcache` (contact reuse's cache, NULL while it is off), so a
consumer regenerates its bindings and deploys a matching set. Its source of record is the patch
stack in the game's repository (`tools/mujoco/patches/`, applied in order to upstream 3.12.0), and
`mjf_buildId()` names the source a library was built from.

## The integration API (`mjfeather.h`)

- **Identity and layout.** `mjf_buildId` identifies the native source and the declared dependency
  pins, dirty source files included; the ID is common to the Linux and Windows libraries, whose
  hashes are recorded separately. `mjf_layout(type, field)` returns the size of `mjModel`, `mjData`,
  `mjOption`, `mjContact`, `mjtNum`, `mjLogMessage` and `mjfRayHit`, and the offsets of selected
  fields, so a consumer checks its mirrors before it allocates or dereferences anything.
- **Timers.** `mjf_enableTimers(1)` installs a monotonic wall clock in milliseconds as `mjcb_time`
  (`CLOCK_MONOTONIC` on Linux, `QueryPerformanceCounter` on Windows). It is opt-in and
  process-global: disabling sets NULL, not a previous callback, and either is done only while every
  simulation is idle. It reads and clears no `mjData`. Upstream's call counts advance even with no
  clock; only the durations stop. The timers nest (STEP holds FORWARD and ADVANCE, POSITION its
  sub-stages, COLLISION its BROAD and NARROW), so sub-stages need not sum to their parent.
- **Queries.** `mjf_raycast` casts one ray with `mj_ray`'s intersection, visibility and static rules
  and returns the hit's distance, point, outward unit normal, geom and body. `mjf_overlapSphere`
  returns every geom whose bounding sphere meets a query sphere, planes as solid halfspaces, and
  the total count even when the caller's buffer is too small. Both keep a geom whose
  `contype | conaffinity` shares a bit with the mask (-1 every collidable geom, 0 none) and skip one
  body. Both test every geom; only a mesh's ray uses a tree, the mesh's own.
- **Heightfield writes and wakes.** `mjf_writeHfieldRows` and `mjf_writeHfieldData` validate the
  whole input (finite, the exact count) before writing, clamp to [0, 1], write only changed samples
  and wake the sleeping islands with a geom that can touch the field over them: the geom's
  footprint in each instance's own frame, the adjacent cells, and a margin (the two geoms' summed
  margins, or the largest explicit pair's if that is larger). Height never rules a wake out, so
  lowered support wakes what it supported. A byte mask on the data's stack, alive only during a
  write with sleep on, keeps two disjoint prints in one row from waking a rack between them. An
  identical or invalid write wakes nothing. `mjf_wakeHfieldRegion` wakes over a rectangle of samples, for raw writes or
  for other datas sharing the model (with no mask, the whole rectangle counts as changed), and
  `mjf_wakeAll` wakes every island. A wake is upstream's `mj_wakeIsland` and `mj_updateSleep`: the
  whole island, already-awake trees untouched, never a velocity or a force. Callers sharing a model
  synchronize writes with stepping, wake every other live data, and keep geom transforms current.
- **Contact reuse.** `mjf_setContactReuse` and `mjf_contactReuseStats`, below.

## The smooth heightfield collider

Upstream treats each heightfield cell as a solid triangular prism, whose internal side faces give
lateral normals even on a flat field. Over the grid interior this collider gives a smooth terrain
normal and at most four contacts per pair, anchored at hull vertices, terrain samples and lattice
crossings, never at clipped witnesses.

- Heights stay upstream's triangulated surface (the diagonal from sample (r, c) to (r+1, c+1)).
  Normals come from a smooth field through the same samples: central-difference gradients at the
  samples, bilinear inside a cell, continuous across cells and exact on any plane, so flat or
  uniformly sloped terrain reduces to the plane functions.
- Polytopes (boxes, meshes): against the triangulated heights the clearance is minimal only at
  lower hull vertices, lattice samples under the lower surface, or crossings of lower edges with the
  lattice's row, column and diagonal lines, so the deepest candidate is exact. Lattice features join
  the support polygon only where the terrain bends upward (a ridge line or a peak sample), a
  property of the terrain alone, so the choice cannot flip while the geom rocks. On a plane the
  contacts are `mjc_PlaneBox`'s.
- Spheres and capsules: the plane functions' points, each against the terrain below it on the smooth
  normals, plus the deepest terrain feature when it is deeper by more than a tenth of the margin. A
  capsule's side also rests on ridge lines and peak samples.

`mjc_SmoothHField` is the collision table's heightfield entry. It hands the side walls, the
underside and every other shape (cylinders, ellipsoids) to upstream's `mjc_ConvexHField`. A geom's
footprint over the field (`mjc_hfieldGeomBox`) is shared by the collider and the regional wake.

## The broadphase and midphase

Each change only skips work, and the contacts of a step are the ones every admitted geom pair would
give (`engine_collision_broadphase_test` checks it against a brute-force pass); no public layout
changes.

- **World-axis broadphase.** Each body is boxed in world axes: the union of its geoms' boxes, each
  the overlap of the geom's bounding sphere and its oriented box as upstream builds them, padded by
  the body's largest margin + gap. The sweep runs along the world axis the box centres spread most
  on. A body whose weld has no dofs (static or mocap) never pairs with another such body, so bodies
  with dofs sweep among themselves and each dof-less box scans only the span of them it covers.
  The pairs sort by a bucket per first body. Flex models keep upstream's broadphase.
- **Node masks.** Per node of a body's tree, the OR of `geom_contype` and of `geom_conaffinity`
  below it; the tree walk skips a node pair when no geom below one can collide with a geom below
  the other. They are read from the live masks, filled the first time a body reaches the midphase
  in a step, and off under `mjcb_contactfilter` and while the visualizer marks active bounding
  volumes (`vis.global.bvactive`).
- **Class-grouped body trees** (compiler). A body's tree splits by collision class first, then at
  the median along the longest axis as upstream does, so the node masks drop a class at one node
  instead of at every leaf. Only the tree's interior changes; the root box, which the ray code
  reads, does not.
- **Covered body pairs.** When a body pair's explicit `<pair>` rows list every geom pair the bitmask
  admits, the walk is skipped (checked each step from the live masks, by marking the listed pairs in
  a bitmap); off under a contact filter or `bvactive`.
- **One-pair collisions.** A geom pair the walk finds is collided directly into a local buffer, and
  its contacts are allocated only when the collision function returns some. A pair whose contact
  bound exceeds `mjMAXCONPAIR` (SDF) keeps `mj_narrowphase`.
- **Explicit rows sphere-first; a resumed exclude scan.** Each `<pair>` row runs its bounding-sphere
  test before the rest of the filter, and the `<exclude>` scan resumes where the previous body pair
  left it.

## Positive-margin convex manifolds

With a positive margin, upstream sends box and mesh pairs through its older perturbed-pose path.
Here the native clipper handles them too: EPA tests the inflated shapes and chooses the candidate
features; where it recovers opposing faces, the original faces are clipped, vertices within the
combined margin kept, each one's signed depth taken along the reference normal, and the inflated
witnesses rebuilt from half of each object's own margin. The recovered normal is carried
explicitly, since near the margin the inflated witnesses can coincide to machine precision. Edges
without matching faces keep the EPA witness. Zero-margin arithmetic, the polygon reduction, the
four-contact mesh cap, the native CCD and multiccd flags and the libccd fallback are upstream's.

## Contact reuse

Off by default. At a few hundred steps a second most touching pairs barely move between steps, so
a pair may replay its last collision instead of running the collision function again.

- **The gate.** Each replayable pair keeps, from its last real collision, the second geom's pose in
  the first's frame and the contacts. It replays while no point of the second geom has moved more
  than the tolerance relative to the first (the shift of its origin plus the rotation's chord at its
  bounding radius), measured from the last real collision, so drift accumulates.
- **The replay.** Each contact is stored as a point on each surface, each in its own geom's frame,
  and a normal and first tangent in the first geom's frame; replayed, its depth is the points'
  separation along the normal and its position their midpoint, and one beyond the margin is dropped.
  Every other contact parameter comes from the live model.
- **What forces a collision.** No entry; the tolerance exceeded; a different margin + gap or geom
  size; a heightfield write within two samples of the pair's footprint; more than eight contacts.
  Plane, sphere and capsule pairs and SDF pairs always collide. `mjfREUSE_EMPTY` also replays pairs
  that found nothing, which can report a new contact up to the tolerance late.
- **State.** The cache belongs to its `mjData`: `mj_resetData` empties it, `mj_copyData` copies it,
  `mjv_copyData` leaves the destination's alone, `mj_deleteData` frees it, and enabling empties it.
  A state set with `mj_setState` is not a reset: a pair replays only while its geoms sit within the
  tolerance of their recorded pose, which is exactly when the replay is valid.

On the game's arena replay, a 0.05 mm tolerance cut collision from 151 to 107 µs a step standing
(170 to 155 lying), with bodies within 1 mm of the run without reuse after a second.

## What the fork does not add

No cache of the previous step's constraint forces (MuJoCo's Newton warm start rebuilds them from
`qacc_warmstart`, so copied forces would be overwritten) and no position-level friction anchors (a
history-dependent friction law that would need its own validated design).

## Tests

- Native (`ctest`): `engine_collision_hfield_test` (the depth against a brute-force oracle on five
  terrains, parity with the plane functions on a flat field, continuity while a geom slides across
  cells, the prisms' fallback), `engine_collision_broadphase_test`, `engine_collision_manifold_test`,
  `engine_collision_reuse_test` and the positive-margin case in `engine_collision_gjk_test`.
- The game's arena replay (`tools/mujoco/replay/` in its repository): recorded GladiatorArena states,
  standing and lying, hashed after stepping; a change meant to be exact keeps the hash on both
  platforms.

## Toolchain notes

Build the Linux library with Ubuntu 22.04's gcc 11.4 and the Windows one with LLVM-MinGW 20260616,
both from one LF checkout so the source ID matches. Build any Python extension against the same
core and toolchain: a GCC-built engine swapped into the official libc++ wheel breaks `MjSpec`'s C++
ABI. The one-line MJZ fix converts filesystem paths to narrow strings before `%s`, which lets the
LLVM-MinGW build compile; it changes neither the simulation nor the ABI.

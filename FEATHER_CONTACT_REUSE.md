# Feather: contact reuse across steps

At a few hundred steps a second, most touching geom pairs barely move between steps: on the arena
replay, 99.6 % of contacting pairs still touch one step later, and 97 % of feet on sand move less
than 0.1 mm. Contact reuse (integration13, off by default) lets such a pair replay its last
collision instead of running the collision function again, as Jolt's body-pair cache, PhysX's
persistent manifolds and Box2D's contact recycling do.

## The gate

Each replayable pair keeps, from its last real collision, g2's pose in g1's frame and the
contacts. A pair replays while no point of g2 has moved more than the tolerance relative to g1:
the shift of g2's origin in g1's frame plus the rotation's chord (`2 sin(angle/2)`, from the trace
of the relative rotation) at g2's bounding radius. The bound is on points, not on Jolt's separate
translation and angle, so a long geom (a foot, a hull) rotates little before it collides again.
The reference is the last real collision, so drift accumulates against the tolerance instead of
resetting each step.

## The replay

Each contact is stored as a point on each surface (its position minus and plus half its distance
along the normal), each in its own geom's frame, and a normal and first tangent in g1's frame.
Replayed, the points and normal are taken back to the world through the geoms' current poses; the
depth is the points' separation along the normal, the position their midpoint. A contact whose
depth now exceeds the margin is dropped, as the collision function would have. Condim, friction,
solref, solimp and the rest come from the live model afterwards, as for any contact.

## What forces a collision

- no entry, or the tolerance exceeded;
- a different margin + gap, or a different size of either geom;
- a heightfield write through `mjf_writeHfieldRows`, `mjf_writeHfieldData` or
  `mjf_wakeHfieldRegion` whose changed samples lie within two samples of the pair's footprint
  (grown by the margin and the tolerance): the collider's normals read the ring around it;
- more than eight contacts (the pair then always collides).

Pairs of two planes, spheres or capsules always collide: their functions cost less than a lookup.
SDF pairs always collide. `mjfREUSE_EMPTY` also replays pairs that found no contact; a contact
that appears within them can then be up to the tolerance late.

## State

The cache belongs to its `mjData` (`contactcache`, a trailing field): `mj_resetData` empties it,
`mj_copyData` copies it (so a copy steps exactly as its source would), `mjv_copyData` leaves the
destination's alone, and `mj_deleteData` frees it. `mjf_setContactReuse(d, 1, tolerance, flags)`
turns it on, or back on, with an empty cache; `(d, 0, ...)` frees it. A state loaded with
`mj_setState` is not a reset: a pair then replays only if its geoms happen to sit within the
tolerance of their recorded pose, which is exactly when the replay is valid. Entries a pass has
not touched since the previous one are dropped when the table fills.

## Measured

The GladiatorArena replay (40 recorded states, statics in 10 m cell bodies, Linux, serial), with
contacts replayed only when found (no `mjfREUSE_EMPTY`):

| Tolerance | Collision per substep, standing | Lying | Largest body drift after 0.1 s (median / max) |
|---|---|---|---|
| off | 151 µs | 170 µs | — |
| 0.025 mm | 113 µs | 163 µs | 0.02 / 0.06 mm |
| 0.05 mm | 107 µs | 155 µs | 0.04 / 0.11 mm |
| 0.1 mm | 98 µs | 148 µs | 0.08 / 0.18 mm |

Over a full second, at 0.05 mm, the bodies stay within 1 mm of the run without reuse (median
0.12 mm); adding 1e-4 to every velocity at the start moves them further (median 0.39 mm, up to
63 mm on a stumbling state). Replays barely rise past 0.1 mm while the drift does: the pairs are
either planted or clearly moving. The replays that matter are feet planted on sand, about 4 µs a
collision.

## Tests

`test/engine/engine_collision_reuse_test.cc`: a replay at the recorded pose reproduces the
collision's contacts; depths follow the anchors; translation or rotation past the tolerance
collides; a heightfield write under the pair forces a collision and one elsewhere does not; a copy
steps bit-identically to its source; reset empties the cache; invalid settings are refused.

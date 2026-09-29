# Feather: smooth heightfield contacts

Convex geoms against a heightfield's top surface. Upstream treats each cell as a solid triangular
prism, whose internal side faces give lateral normals even on a flat field. Over the grid interior
this collider gives a smooth terrain normal and at most four contacts per pair, anchored at hull
vertices, terrain samples and lattice crossings, never at clipped witnesses.

- Heights stay upstream's triangulated surface (the diagonal from sample (r, c) to (r+1, c+1)).
  Normals come from a smooth field through the same samples: central-difference gradients at the
  samples, bilinear inside a cell. They are continuous across cells and exact on any plane, so
  flat or uniformly sloped terrain reduces to the plane functions.
- Polytopes (boxes, meshes): against the triangulated heights the clearance is minimal only at
  lower hull vertices, lattice samples under the lower surface, or crossings of lower edges with
  the lattice's row, column and diagonal lines, so the deepest candidate is exact. Lattice
  features join the support polygon only where the terrain bends upward (a ridge line or a peak
  sample), a property of the terrain alone, so the choice cannot flip while the geom rocks. On a
  plane the contacts are `mjc_PlaneBox`'s vertices.
- Spheres, capsules, ellipsoids, cylinders: the plane functions' points, each against the terrain
  below it on the smooth normals, plus the deepest terrain feature when it is deeper by more than
  a tenth of the margin. A capsule's side also rests on ridge lines and peak samples.

## Choosing the collider

`mjModel.hfield_collider` holds one byte per heightfield: 0, the compiled default, is this
collider over the interior; 1 is upstream's prisms throughout. Set and read it with
`mjf_setHfieldCollider(m, id, collider)` and `mjf_hfieldCollider(m, id)`, while no collision runs
on the model. `mjc_SmoothHField` is the collision table's heightfield entry; it hands the side
walls, the underside, unsupported geoms and fields set to 1 to upstream's `mjc_ConvexHField`. The
footprint of a geom over the field (`mjc_hfieldGeomBox`) is shared by the collider and the
regional wake ([FEATHER_REGION_WAKE.md](FEATHER_REGION_WAKE.md)).

Integration11 deleted the earlier paths: the polyhedral top-surface collider (`mjc_PolyHField`),
face clipping, the min/max height pyramid and its three model arrays, and the process-global
collider switch.

## Tests

Run `python/mujoco/hfield_test.py` and `python/mujoco/hfield_wake_test.py` against bindings linked
to this build, plus the upstream collision, ray and sleep tests. The ray tests need the normal
CTest `MUJOCO_PLUGIN_DIR` environment to find the SDF plugin.

## Toolchain notes

Use matching source and headers for Linux, Windows, and Python builds. Swapping a GCC/libstdc++
engine into the official libc++ Python wheel is unsafe for `MjSpec`; rebuild Python extensions
with the engine's toolchain. Plain C ABI compatibility does not make the C++ objects behind
MjSpec compatible.

The one-line MJZ warning fix converts filesystem paths to narrow strings before passing them to
`%s`, which permits the LLVM-MinGW Windows build. It changes neither the simulation nor its ABI.

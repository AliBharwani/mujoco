# Feather: polyhedral heightfield contacts

This 3.9.0 fork adds an interior top-surface collision path for boxes and meshes
with compiled convex polygons. It fixes lateral contacts against the internal
walls of the triangular prisms used by the original heightfield collider.

`mjc_PolyHField` transforms the polyhedron into the heightfield frame, intersects
its downward-facing hull triangles with the XY footprints of terrain triangles,
and computes each witness's signed top-plane distance, midpoint, and normal from
that same plane. Contact reduction retains at most four spatially separated
witnesses per coplanar patch, including the deepest, subject to the existing
`mjMAXCONPAIR` cap. Geometry is never modified, so shared models remain usable.

The original collider handles other primitive types, meshes without compiled
convex polygons, and a whole pair whenever the polyhedron reaches the field's
outer walls or underside. Thus the top-normal guarantee applies to the interior
box/convex-mesh path. This is a discrete contact-generation fix, not swept CCD.
No public header or struct layout changes; C# bindings need no regeneration.

The 3.9 integration2 build first attempts an uncached whole-polyhedron height
rejection. It scans samples over the footprint and bounds both height and slope
so contact margin remains a normal-distance allowance. It stops scanning when
the bound cannot reject, preserving the original contact traversal and reduction.
Elevated rack geometry avoids repeated per-facet terrain-triangle construction;
geometry actually in contact still uses the full top-surface path. There is no
height cache or additional write/invalidation contract. Tiny cell areas and
unsupported margins retain the reference path.

Run `python/mujoco/hfield_test.py` against bindings linked to this build. It covers
flat cell seams, both CCD flag settings, three resolutions, margins, five-second
passive drift, slopes, ridges/valleys, arbitrary orientations, rotated fields,
dense nonplanar terrain, and preservation of outer-wall/underside contacts.
Six of the original eight tests fail on stock 3.9. Two additional regressions
cover normal-distance margin above a steep ridge and live writes after a rejected
pair. Also run the
upstream collision, ray, and sleep tests. The ray tests require the normal CTest
`MUJOCO_PLUGIN_DIR` environment to find the SDF plugin.

Use matching source and headers for Linux, Windows, and Python builds. Swapping
a GCC/libstdc++ engine into the official libc++ Python wheel is unsafe for
`MjSpec`; rebuild Python extensions with the engine's toolchain. Plain C ABI
compatibility does not make the C++ objects behind MjSpec compatible.

The accompanying one-line MJZ warning fix converts filesystem paths to narrow
strings before passing them to `%s`, which permits the LLVM-MinGW Windows build.
It changes neither the simulation nor its ABI.

Application acceptance remains a separate test: compare the original frozen
flat-heightfield stance probe to the box floor with the same policy and contact
parameters. Passive geometry tests do not establish closed-loop policy behavior.

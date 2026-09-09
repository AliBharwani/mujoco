# Heightfield min/max pyramid (3.12 integration3)

`mjModel` owns `hfield_mipmin`, `hfield_mipmax` and `hfield_mipvalid` through
the normal model-buffer/X-macro lifecycle. No process-global pointer cache or
per-step heap allocation. Python introspection and C# layouts are regenerated.
MJB files from the previous fork layout must be recompiled from MJCF.

Leaves cover 2x2 cells and all nine corner samples; each next level combines
four children, rounding odd dimensions up. Packed levels start at `hfield_adr`
and fit inside `nrow*ncol` entries per field (including rectangular/skinny grids).
Both min and max are necessary: their range bounds terrain slope for a
conservative normal-distance contact margin. The minimum of a lower hull
triangle's plane over a tile bounds its local height; ill-conditioned XY
projections fall back to its vertex minimum. Surviving cells retain the original
row-major clipping and contact-reduction order. Tiny footprints bypass traversal.

Compilation builds the pyramid, but leaves it **disabled**: upstream permits raw
`hfield_data` writes, and C cannot observe those writes automatically. Call
`mjf_enableHfieldMip(m, id, 1)` after initial terrain setup to refresh and opt in.
While enabled, write through `mjf_writeHfieldData` or `mjf_writeHfieldRows`. If
using raw writes, refresh with `mjf_enableHfieldMip(..., 1)` again before any
collision, or disable the cache first. Re-enabling validates finite heights.

`mjf_writeHfieldRows(m, d, id, row_start, row_count, values, count)` accepts
contiguous rows (`count == row_count*ncol`). It validates the entire input before
mutation, clamps values to [0,1], and refits only tiles affected by changed
samples, including the adjacent cells across tile boundaries. Identical values
do not refit or wake. As with every model edit, synchronize **all** model readers;
the write is validation-atomic, not a concurrency primitive. In integration3 the
legacy wake-all behavior is retained; regional wakes are a separate next change.

`hfield_mip_test.py` covers raw-write compatibility, odd/rectangular/skinny grids,
tile-edge edits, decreasing maxima, validation atomicity, copy/MJB lifetime,
rotated fields, randomized boxes/meshes/margins and exact trajectory parity.
The actual-ankle/racks/slab 900-step replay also matches integration2 exactly.
The extra model buffers change the MJB hash; that is explicitly allowed by the
replay flag, while all options and every dynamics/contact hash must still match.

The pyramid cannot reject tiles carrying real contacts. Synthetic loaded slabs
benefit; standing small feet/racks show little improvement. The requested Unity
<100 us / density-ratio target remains an external acceptance test, not a claim
established by this implementation.

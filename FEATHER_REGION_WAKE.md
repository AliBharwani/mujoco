# Heightfield region wakes (3.12 integration4)

Full and partial-row writes now wake sleeping islands only where actual changed
samples can affect a geom. Validate the complete input before writing, track the
changed samples, then use current geom AABBs transformed into each hfield geom's
local frame. Include adjacent cells and conservative contact margins, including
explicit-pair/override margins. A single asset can have several translated or
rotated instances; all are considered. Z is deliberately not used to reject a
potential wake, so a lowered support surface also wakes the supported body.

The write path's temporary byte mask distinguishes disjoint prints in the same
row: a rack between them is not woken merely by their shared bounding rectangle.
It uses mjData's stack and exists only during a synchronized write with sleep
enabled. With sleep disabled no mask is allocated. This is separate from mip
refitting (which can conservatively refit the dirty bounding rectangle).

Waking calls upstream `mj_wakeIsland`, then `mj_updateSleep`: the complete
sleeping island is woken, including connected trees. Already-awake trees are
left alone; continuous nearby sand updates do not reset their sleep countdown.
No position, velocity, force or global sleep policy is changed. Natural contact
wakes remain upstream's responsibility.

`mjf_wakeHfieldRegion(m, d, id, row_start, row_count, col_start, col_count)` is a
public rectangular sample-region wake for raw writes or other datas sharing the
same model. It has no dirty mask, so conservatively treats that whole rectangle
as changed. It returns the number of trees woken, or -1 for invalid arguments.
No-op/invalid native writes cause no wake. All shared-model readers must be idle
during writes/refits, and all affected datas must be woken before stepping again.
The supplied data's geom transforms must be current (`mj_forward` after manual
pose changes). Normal stepping already maintains them for sleeping bodies.

Unity owns enabling sleep and assigning tree policies: humanoids NEVER, passive
racks AUTO. Never assume this leaves the humanoid's physical inputs identical:
sleeping nearby props stop being dynamic until woken, so gameplay still needs
acceptance even though the humanoid itself stays awake. Turning sleep on is not
performed by this API or in Feather's training environment.

`hfield_wake_test.py` covers one print/one rack, disjoint prints, no-op/invalid/far
writes, adjacent cells, translated/rotated fields, multiple asset instances,
equality-connected sleeping islands, shared-data ownership and upstream contact
wakes. The ordinary sleep-off hfield replay must remain exact.

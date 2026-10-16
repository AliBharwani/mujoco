# Feather: broadphase and midphase

The midphase walks the geom trees of two bodies whose boxes overlap and hands each geom pair whose
boxes meet to the narrowphase. Every change below is bit-exact: the arena replays' trajectory
hashes match the previous core. None changes a public layout.

## Node masks (integration9)

Per node of a body's tree, the OR of `geom_contype` and the OR of `geom_conaffinity` over the geoms
below it. `mj_collideTree` skips a node pair when no geom below one node can collide with a geom
below the other, so it never tests boxes whose every leaf pair the bitmask rejects. They are off
under `mjcb_contactfilter`, which replaces the bitmask, and while the visualizer marks active
bounding volumes, which a pruned pair would leave unmarked. They are read from the live geom masks,
so a mask edited at runtime holds from the next step.

## Integration12

- **Class-grouped body trees (compiler).** `mjCBody::ComputeBVH` builds each body's tree by
  collision class (`contype`, `conaffinity`) first: while a node's geoms span more than one class,
  the list of classes splits in half; a single class splits at the median along the longest axis,
  as upstream does. The node masks then drop a class the other body cannot touch at one node
  instead of at every leaf. The classes are the compiled ones: a mask edited at runtime only
  prunes less. Only the tree's interior changes. The root box, which the ray code reads, is the
  same, and a body pair's contacts are sorted after the walk, so no result depends on the walk
  order. Mesh and flex trees are unchanged.
- **Covered body pairs.** When the explicit `<pair>` rows merged for a body pair list every geom
  pair of theirs that the bitmask lets collide, every candidate the walk could find is one
  `filterCollisionPair` drops as a duplicate, so the walk is skipped (`coveredByPairs`, checked
  each step from the live masks; off under a contact filter or the bounding-volume visualizer).
  The queued geom pairs are still flushed where the walk would have flushed them: merged, the
  batches crossed `mj_narrowphase`'s 16-pair chunk and paid the thread pool's wake-up.
- **One-pair collisions.** A geom pair the walk finds is collided directly (`mj_collideGeoms`)
  into a local buffer, and its contacts are allocated and filled (`setGeomContacts`, shared with
  `mj_narrowphase`) only when the collision function returns some; most return none. A pair
  whose contact bound exceeds `mjMAXCONPAIR` (SDF) keeps `mj_narrowphase`.
- **Lazy node masks.** A body's masks are filled the first time it reaches the midphase in a step,
  not for every body.
- **Resumed exclude scan.** The `<exclude>` scan resumes where the previous body pair left it; both
  lists sort by signature.

Measured on the GladiatorArena replay (40 recorded states, the arena model with its statics in
10 m cell bodies, Linux, 2 island threads, integration11 → 12): standing, narrowphase 146 → 122
µs and step 468 → 436 µs (−6.8 %); lying, 171 → 143 and 503 → 473 (−5.9 %). Serial: −6 % and
−8.4 %.

## Integration13

- **World-axis broadphase.** Upstream boxes every geom in a frame from the covariance of all geom
  centres (its eigenvectors, then a rotation per geom), sweeps one axis of it, and sorts the pairs.
  Integration13 boxes each body in world axes: the union of its geoms' boxes, each the overlap of
  the geom's bounding sphere and its oriented box as upstream builds them, padded by the body's
  largest margin + gap. The sweep runs along the world axis the box centres spread most on. A body
  whose weld has no dofs (static or mocap) never pairs with another such body, so the bodies with
  dofs sweep among themselves and each dof-less box scans only the span of them it covers. The
  pairs sort by a bucket per first body. Any boxes that hold every geom with its margin give the
  same contacts: a body pair out of reach finds none in the midphase, and the pairs reach it sorted
  by signature. Flex models keep upstream's broadphase (its per-vertex boxes).
- **Explicit rows sphere-first.** Each `<pair>` row runs its bounding-sphere test on its own, with
  `filterCollisionPair`'s arithmetic, before the rest of the filter: on the arena 95 % of the rows
  stop there.
- **Covered pairs by bitmap.** `coveredByPairs` marks the listed geom pairs in a bitmap and then
  looks for an admitted, unmarked one, instead of scanning the rows for every geom pair.

Measured on the arena replay (Linux, serial, instrumented copy of the driver): broadphase 47 → 17
µs a substep standing (49 → 18 lying), the explicit rows 9.7 → 7.6 µs, the covered-pair checks
8.7 → 4.8 µs. Pairs reaching the midphase: 112.6 → 115.3 standing, 158.4 → 158.8 lying.

Not taken: skipping groups of explicit rows by a body-level bounding sphere (it dropped only 30 %
of the rows and cost more than it saved), per-body root boxes of the trees for the broadphase
(cheaper boxes, but 20 more body pairs to walk), and caching the static bodies' boxes across steps.
[Contact reuse](FEATHER_CONTACT_REUSE.md), the integration13 option that is not bit-exact, has its
own page.

Not taken: skipping explicit pairs whose body pair the broadphase did not report. A check that
stays correct when masks or margins are edited at runtime costs about what the sphere test it
replaces costs, and a cached one goes stale.

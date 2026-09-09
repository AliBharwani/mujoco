# Positive-margin convex manifolds (3.12 integration6)

Upstream 3.12 already has native EPA face recovery and polygon clipping for
box/mesh pairs. Positive margins force those pairs through the older perturbed
pose/repeated-EPA path. This change extends the existing native clipper to them;
it does not install another collision engine or introduce contact history.

EPA still tests the inflated shapes and chooses candidate features. Where it
recovers opposing faces, clip the original physical faces, retain vertices within
the combined contact margin, and calculate each vertex's signed depth along the
reference normal. Reconstruct the inflated witnesses using half of each CCD
object's own margin. The existing caller restores physical distance and midpoint.
Integration6 also carries the recovered reference normal explicitly in the
internal CCD result. Near the margin the inflated witnesses can coincide to
machine precision, so deriving the normal from their difference loses its
direction. A binary-exact tilted-box mesh regression covers that boundary.
Zero-margin arithmetic and the existing polygon reduction are preserved. Mesh
pairs retain the four-contact cap, box-box the upstream eight-contact cap.

For positive-margin edges without matching faces, keep the original EPA witness;
those inflated edge points are not physical edge endpoints. No fabricated face
or stale force is substituted. Native CCD/multiccd flags still select the upstream
fallbacks. GJK's distinction between smooth inflated and discrete support shapes
is unchanged. No solver/friction settings or public layouts change in this item.

Tests cover separated/penetrating parallel faces, both shape orders, translation,
yaw, tilted-face physical witnesses, margin exclusion, edge fallback, flags,
polygon reduction, zero margins, and unequal low-level CCD object margins.
Actual-ankle quality probes are passive native fixtures, not Unity acceptance.

This candidate also corrects regional waking to use the SUM of the two geom
margins, as 3.12's collision driver does. A regression requires the sum to wake a
nearby sleeping rack. Integration4 is preserved for provenance but integration6
supersedes integrations4/5 for deployment.

# Contact history and friction-anchor evaluation (2026-09-09)

The requested efc_force cache is not implemented. In the Newton path,
engine_forward.c warmstart() evaluates qacc_warmstart by rebuilding J*qacc-aref
and calling mj_constraintUpdate. engine_solver.c mj_solPrimal repeats the
constraint update before certification/iterations. Existing efc_force values are
not the primal initial state. Copying previous contact forces into that buffer
would be overwritten; retaining them after the solve would violate the solved
force/acceleration relationship.

A controlled fixed-state probe clears efc_force and repeats mj_forward: forces
are identical. When the exact fixed-state solution is supplied in qacc_warmstart,
clearing efc_force and using zero Newton iterations still reconstructs its
nonzero support force. This does not claim that every newly found contact has
positive force: geometry, load redistribution and the selected optimum determine
that. Nor does it claim contact identity cannot be useful for diagnostics.

The reported one-iteration convergence makes a new initial-guess cache a weak
candidate for fixing resting load blinks. First distinguish geometric absence,
present-but-inactive margin/gap contacts, and present contacts with nearly zero
normal load. Record geom pairs, signed distance, includemargin/exclude/efc_address,
normal force, awake state and the load threshold. Use the manifold and regional
wake candidates before attributing remaining blinks to missing force history.
No blink acceptance target is claimed closed by this reasoning or the passive
ankle fixture. The Unity per-rack probe remains required.

Position-level friction anchors are also not implemented. They add a new
history-dependent tangential constitutive law, beyond a warm-start optimization.
A principled design would need material-frame anchors, transported tangent bases,
stick/slip projection and release, per-data state with reset/copy/serialization,
sleep/wake and contact-removal rules, rotating/moving supports, friction-cone and
energy tests, and forward/inverse/derivative semantics. It must preserve sliding
above the friction limit and cannot pin a foot in world space. That work requires
an explicit contact-model experiment with matched training/runtime validation.

MuJoCo's existing noslip postprocessor is an available comparison, not evidence
that anchors are needed or a default changed by this batch. The supplied quality
probe includes noslip=0/10, zero/sub-limit/above-limit lateral loads and box/hfield
floors, with all options and source identities recorded. Native passive results
must not be presented as the closed-loop game's creep or blink acceptance.

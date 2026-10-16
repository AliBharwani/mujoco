# Feather's two engine laws: the speed-shaped brake and the smooth passive friction

Two small changes to MuJoCo's dynamics, made for Feather's human-joint phases (P3a and P5). Both are off by default, and
off they are upstream's operation for operation: the same model gives the same bytes with or without them.

## The speed-shaped brake (`biasprm[3]`, Feather P3a)

An affine-bias actuator's bias is `const + kp*length + kv*velocity`. With `biasprm[3] = vsat > 0`, the velocity in that
term is clipped to `[-vsat, vsat]`, so a position servo's damping acts as upstream up to `vsat` and stays at `kv*vsat`
beyond it. `mjd_actuator_vel` gives the term the slope `kv` inside the band and 0 outside, the exact derivative.
`vsat <= 0` is upstream's affine bias.

- It also clips the kv term of a `<general>` PID-style actuator if `biasprm[3]` is set there. `<pid>` and
  `mjs_setToPID` never set it.
- `dampratio` keeps `biasprm[3]` when it converts `biasprm[2]`.
- A stiff, light servo can flip across the band: beyond `vsat` the brake is a constant torque applied explicitly, so
  when `timestep*kv*gear^2/I` exceeds about 2, a velocity just past `vsat` can land past `-vsat` the next step, and keep
  alternating. Upstream's own force clamp has the same mode above `cap/kv`; the brake lowers its onset to `vsat`.

## The smooth passive friction (a negative `dof_frictionloss`, Feather P5)

A negative `dof_frictionloss = -F` is a passive friction of size `F`: the force `-F*tanh(v/v0)`, `v0 = 0.05` (rad/s, m/s
on a slide dof; `mjPASSIVE_FRICTION_V0`), added to `qfrc_damper` with the dof dampers. It adds no constraint row
(`mj_instantiateFriction` skips values `<= 0`). Its slope (`mj_frictionSlope`) enters `mjd_passive_vel` (implicit and
implicitfast), `mj_EulerSkip`'s implicit damping and the discrete inverse (`mj_discreteAcc`, `mjENBL_INVDISCRETE`).

- It is off with `mjDSBL_DAMPER` or `mjDSBL_FRICTIONLOSS` (the constraint friction only with the latter).
- Ball and free joints get it per axis, as upstream's per-dof frictionloss. A negative tendon frictionloss is ignored,
  before and after.
- The XML compiler accepts a negative frictionloss and `spec.to_xml` writes it back. Upstream reads a negative value as a
  constraint that adds energy, so no valid upstream model has one.
- It has no stiction: a dof under a steady torque `tau < F` creeps at `v0*atanh(tau/F)` instead of sticking, so a body
  can take longer to fall asleep than under the constraint friction.
- Keep one step's friction impulse over the dof's inertia, `timestep*F/I`, under about `5*v0` (under implicitfast; `2*v0`
  if the friction were explicit), or a dof near rest chatters, its velocity flipping sign every step.

## Tests

`test/engine/`: `ActuatorTest.SpeedShapedBrake`, `ActuatorTest.SpeedShapedBrakeDampRatio`,
`DerivativeTest.SpeedShapedBrakeDerivative`; `PassiveTest.SmoothFrictionForce`, `PassiveTest.SmoothFrictionDisableFlags`,
`DerivativeTest.SmoothFrictionDerivative`, `CoreConstraintTest.SmoothFrictionNoConstraintRow`,
`ImplicitIntegratorTest.SmoothFrictionEulerImplicitEquivalent`, `ImplicitIntegratorTest.SmoothFrictionStiffComesToRest`,
`InverseTest.DiscreteInverseSmoothFriction`.

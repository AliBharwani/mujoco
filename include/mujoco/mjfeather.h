// mjfeather.h — Feather fork additions to the MuJoCo C API.
//
// The spatial queries, heightfield writes and wakes, collider selection, timers and build
// identity the Feather/Unity port calls. Kept in this header and the fork's own source files
// (engine_query.c, engine_feather.c, engine_collision_hfield.c) under the `mjf_` namespace
// so the fork rebases cleanly onto upstream MuJoCo tags. These reuse MuJoCo's own geom
// state, so query results stay consistent with the simulation. Exposed with
// MJAPI so the symbols are visible in libmujoco for C# P/Invoke.
//
// Requires up-to-date kinematics: call after mj_step / mj_forward so
// geom_xpos (and the ray's collision state) are current.

#ifndef MUJOCO_MJFEATHER_H_
#define MUJOCO_MJFEATHER_H_

#include <mujoco/mjexport.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjdata.h>

#ifdef __cplusplus
extern "C" {
#endif

// Stable source identity and layout introspection; safe before allocating a model.
// layout(type, NULL) returns sizeof(type); a named field returns offsetof(type, field).
// Supported types: mjModel, mjData, mjOption, mjContact, mjtNum. Unknown => -1.
MJAPI const char* mjf_buildId(void);
MJAPI int mjf_layout(const char* type, const char* field);

// Opt-in process-global monotonic wall clock in milliseconds for mjData.timer.
// Nonzero replaces mjcb_time (including Python callbacks); zero sets it to NULL,
// not the previous callback. Call only while all simulations/workers are idle.
// Does not clear timers: durations stop accumulating when disabled, but upstream
// timer call counts still advance. Reading/clearing mjData.timer is caller-owned.
MJAPI void mjf_enableTimers(int enable);

// Convex collider of heightfield `id`, stored in the model (hfield_collider): 0 (the compiled
// default) smooth normals and at most four feature-anchored contacts per pair over the interior,
// with upstream's solid prisms for the side walls and underside; 1 upstream's prisms throughout.
// Call while no collision runs on this model. Returns the previous collider, or -1 (no change)
// for invalid input; the getter returns -1 for an invalid id.
MJAPI int mjf_setHfieldCollider(mjModel* m, int id, int collider);
MJAPI int mjf_hfieldCollider(const mjModel* m, int id);

// Contact reuse across steps, per mjData, off by default. A geom pair whose relative pose has
// moved no more than `tolerance` (m: no point of either geom has moved further relative to the
// other) since its last collision replays that collision's contacts instead of colliding again:
// each contact's depth is re-measured between a point anchored on each geom, along a normal fixed
// to the first, and a contact beyond the margin is dropped. Changed margins or sizes, and
// heightfield writes through mjf_writeHfield* / mjf_wakeHfieldRegion over a pair's footprint,
// force a new collision. Replayed pairs involve a heightfield, box, mesh, ellipsoid or cylinder;
// plane, sphere and capsule pairs always collide. mjfREUSE_EMPTY also replays pairs that found no
// contact, which can report a new contact up to `tolerance` late. The cache is part of the data's
// state: mj_resetData empties it, mj_copyData copies it, mj_deleteData frees it. Enabling
// (again) empties it and zeroes the statistics. Returns 0, or -1 for invalid input.
#define mjfREUSE_EMPTY 1
MJAPI int mjf_setContactReuse(mjData* d, int enable, mjtNum tolerance, int flags);

// Contact reuse counters since it was enabled: pair lookups, replays, replays of no contact,
// collisions recorded. Returns 0, or -1 while reuse is off.
MJAPI int mjf_contactReuseStats(const mjData* d, long long stats[4]);

// Wake all sleeping islands without changing positions, velocities or applied forces.
MJAPI int mjf_wakeAll(const mjModel* m, mjData* d);

// Validate the whole input, clamp to [0,1], write only changes and wake this data's sleeping
// islands with a geom that can touch the field over the change. Shared-model callers must also
// wake their other datas via mjf_wakeHfieldRegion (or mjf_wakeAll for a conservative fallback).
// Returns -1 for invalid input (no writes), 0 for identical data, 1 for a change.
MJAPI int mjf_writeHfieldData(mjModel* m, mjData* d, int id,
                             const float* values, int count);

// Contiguous row update. values contains row_count*ncol floats, starting at
// row_start. Validates before any write; identical rows do not wake or refit.
// Returns -1 for invalid input, 0 for no change, 1 otherwise. All model users
// must be idle during writes. Shared-model callers must wake their other datas.
MJAPI int mjf_writeHfieldRows(mjModel* m, mjData* d, int id, int row_start,
                             int row_count, const float* values, int count);

// Wake sleeping islands over an edited rectangle of SAMPLES (not cells), with
// adjacent-cell and contact-margin expansion, for every geom using this hfield.
// Use after synchronized raw writes or for other datas sharing a written model.
// Kinematics must be current. Returns number of trees woken, or -1 for invalid
// input. Already-awake trees and their sleep counters are left unchanged.
MJAPI int mjf_wakeHfieldRegion(const mjModel* m, mjData* d, int id,
                              int row_start, int row_count, int col_start, int col_count);

// World-space velocities [angular xyz, linear xyz], body or body-frame origin.
// Output shape: [ndata, nbody, 6], double precision. No stepping or allocations.
// Returns -1 for invalid input without writes, otherwise 0.
MJAPI int mjf_bodyVelocities(const mjModel* m, mjData* const* datas, int ndata,
                             const int* bodies, int nbody, int object_type, mjtNum* out);

// Training substeps with post-step joint/root velocity clamps and one-shot linear
// body force on the first substep. Preserve persistent forces; never subtract the
// impulse after MuJoCo auto-resets. Consume BADQPOS/VEL/ACC warnings per substep.
// Returns -1 for invalid arguments, otherwise whether any divergence was consumed.
MJAPI int mjf_stepGuarded(const mjModel* m, mjData* d, int nstep,
                          const int* joint_dofs, const mjtNum* limits, int njoint,
                          int root_dof, mjtNum root_cap, int impulse_body,
                          const mjtNum* impulse_force);

// Result of mjf_raycast. Doubles first for a clean, blittable C#-marshalable layout.
typedef struct mjfRayHit_ {
  mjtNum dist;       // distance from origin to hit along the (normalized) ray
  mjtNum point[3];   // world-space hit point
  int    geom;       // hit geom id (-1 on miss)
  int    body;       // body id of the hit geom (-1 on miss)
} mjfRayHit;

// Cast a ray from origin along dir (need NOT be normalized). Thin wrapper over
// mj_ray (uses the engine's own ray-vs-geom intersection). bodyexclude: skip
// that body (-1 = none). include_static: include static geoms. Writes *out and
// returns 1 on hit, 0 on miss (out is zeroed/(-1) on miss).
MJAPI int mjf_raycast(const mjModel* m, const mjData* d,
                      const mjtNum origin[3], const mjtNum dir[3],
                      int bodyexclude, mjtBool include_static, mjfRayHit* out);

// Broadphase sphere overlap: writes the ids of up to `maxgeoms` geoms whose
// bounding sphere intersects the query sphere into geoms[], and returns the
// TOTAL number found (a return > maxgeoms means the output was truncated).
// bodyexclude: skip that body (-1 = none). This is the broadphase (uses
// geom_rbound); it conservatively over-reports non-spherical geoms and ignores
// planes (rbound 0). Exact narrowphase can be layered on later.
MJAPI int mjf_overlapSphere(const mjModel* m, const mjData* d,
                            const mjtNum center[3], mjtNum radius,
                            int bodyexclude, int* geoms, int maxgeoms);

// Gameplay variants: include geoms whose (contype | conaffinity) intersects mask.
// mask=-1 includes all collidable geoms; mask=0 includes none. Disabled collision
// geoms are excluded. Ray visibility/static rules remain those of mj_ray.
MJAPI int mjf_raycastFiltered(const mjModel* m, const mjData* d,
                              const mjtNum origin[3], const mjtNum dir[3],
                              int bodyexclude, mjtBool include_static, int mask,
                              mjfRayHit* out);
// Includes planes as solid halfspaces, once each; returns TOTAL candidates.
MJAPI int mjf_overlapSphereFiltered(const mjModel* m, const mjData* d,
                                    const mjtNum center[3], mjtNum radius,
                                    int bodyexclude, int mask, int* geoms, int maxgeoms);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_MJFEATHER_H_

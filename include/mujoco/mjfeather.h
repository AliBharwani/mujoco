// mjfeather.h — Feather fork additions to the MuJoCo C API.
//
// The spatial queries, heightfield writes and wakes, contact reuse, timers and build
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
// Supported types: mjModel, mjData, mjOption, mjContact, mjtNum, mjLogMessage (upstream's, which
// the bindings mirror by hand) and mjfRayHit. Unknown => -1.
MJAPI const char* mjf_buildId(void);
MJAPI int mjf_layout(const char* type, const char* field);

// Opt-in process-global monotonic wall clock in milliseconds for mjData.timer.
// Nonzero replaces mjcb_time (including Python callbacks); zero sets it to NULL,
// not the previous callback. Call only while all simulations/workers are idle.
// Does not clear timers: durations stop accumulating when disabled, but upstream
// timer call counts still advance. Reading/clearing mjData.timer is caller-owned.
MJAPI void mjf_enableTimers(int enable);

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
// row_start. Validates before any write; identical rows do not wake.
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

// The queries' mask keeps a geom whose (contype | conaffinity) shares a bit with it: -1 keeps
// every collidable geom, 0 none. bodyexclude skips that body (-1: none). Both test every geom;
// only a mesh's ray uses a tree (the mesh's own).

// Result of mjf_raycast. Doubles first for a blittable C# mirror, which mjf_layout checks.
typedef struct mjfRayHit_ {
  mjtNum dist;       // distance from origin to hit along the (normalized) ray
  mjtNum point[3];   // world-space hit point
  mjtNum normal[3];  // the hit surface's outward unit normal, world frame
  int    geom;       // hit geom id (-1 on miss)
  int    body;       // body id of the hit geom (-1 on miss)
} mjfRayHit;

// Cast a ray from origin along dir (need NOT be normalized) with MuJoCo's own ray-vs-geom
// intersection (mj_ray's visibility and static rules). include_static: include static geoms.
// Writes *out and returns 1 on a hit, 0 on a miss or invalid input (out zeroed, ids -1).
MJAPI int mjf_raycast(const mjModel* m, const mjData* d,
                      const mjtNum origin[3], const mjtNum dir[3],
                      int bodyexclude, mjtBool include_static, int mask, mjfRayHit* out);

// Sphere overlap against each geom's bounding sphere (geom_rbound), planes taken as solid
// halfspaces: writes the ids of up to `maxgeoms` geoms into geoms[] and returns the TOTAL found
// (more than maxgeoms means the output was truncated). Conservative: a non-spherical geom is
// reported when its bounding sphere overlaps. Invalid input finds nothing.
MJAPI int mjf_overlapSphere(const mjModel* m, const mjData* d,
                            const mjtNum center[3], mjtNum radius,
                            int bodyexclude, int mask, int* geoms, int maxgeoms);

// FEATHER training (integration16): the helpers feather's trainer and tests call, which
// integration14 trimmed from Unity's core, restored as they were on feather's integration6 line
// (engine_feather.c, engine_query.c). Unity calls none of them, and nothing in the engine does:
// a program that never calls them steps exactly as on integration15.

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

// mjfRayHit as it was before integration14 added the normal: the result of mjf_raycastFiltered.
typedef struct mjfRayHitNoNormal_ {
  mjtNum dist;       // distance from origin to hit along the (normalized) ray
  mjtNum point[3];   // world-space hit point
  int    geom;       // hit geom id (-1 on miss)
  int    body;       // body id of the hit geom (-1 on miss)
} mjfRayHitNoNormal;

// mjf_raycast and mjf_overlapSphere under their names before integration14 (the gameplay
// "Filtered" pair, whose mask both now take): the same results, the ray's without its normal.
MJAPI int mjf_raycastFiltered(const mjModel* m, const mjData* d,
                              const mjtNum origin[3], const mjtNum dir[3],
                              int bodyexclude, mjtBool include_static, int mask,
                              mjfRayHitNoNormal* out);
MJAPI int mjf_overlapSphereFiltered(const mjModel* m, const mjData* d,
                                    const mjtNum center[3], mjtNum radius,
                                    int bodyexclude, int mask, int* geoms, int maxgeoms);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_MJFEATHER_H_

// mjfeather.h — Feather fork additions to the MuJoCo C API.
//
// Native spatial-query layer for the Feather/Unity port. Kept in a SEPARATE
// header + source file (engine_query.c) under the `mjf_` namespace so the fork
// rebases cleanly onto upstream MuJoCo tags. These reuse MuJoCo's own geom
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

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_MJFEATHER_H_

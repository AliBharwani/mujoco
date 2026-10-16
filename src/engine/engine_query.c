// engine_query.c — Feather fork: native spatial-query layer (see mjfeather.h).
//
// Reuses MuJoCo's own ray intersection (mj_rayMask) and geom state (geom_xpos, geom_xmat,
// geom_rbound) so query results are consistent with the simulation. Both queries test every geom;
// only a mesh's ray uses a tree (the mesh's own). Kept in a separate file for clean rebases onto
// upstream MuJoCo.

#include <math.h>
#include <mujoco/mjfeather.h>
#include "engine/engine_ray.h"
#include <mujoco/mujoco.h>   // mju_normalize3, mju_zero3, mju_copy3

// Invalid input and a zero-length ray miss.
int mjf_raycast(const mjModel* m, const mjData* d, const mjtNum origin[3], const mjtNum dir[3],
                int bodyexclude, mjtBool include_static, int mask, mjfRayHit* out) {
  if (out) {
    out->dist = -1;
    mju_zero3(out->point);
    mju_zero3(out->normal);
    out->geom = out->body = -1;
  }
  if (!m || !d || !origin || !dir) return 0;
  for (int j = 0; j < 3; j++) if (!isfinite(origin[j]) || !isfinite(dir[j])) return 0;
  mjtNum v[3] = {dir[0], dir[1], dir[2]};
  if (mju_normalize3(v) <= mjMINVAL) return 0;
  int geom = -1;
  mjtNum normal[3];
  mjtNum dist = mj_rayMask(m, d, origin, v, NULL, include_static, bodyexclude, &geom, normal, &mask);
  if (geom < 0 || dist < 0) return 0;
  if (out) {
    out->dist = dist;
    mju_addScl3(out->point, origin, v, dist);
    mju_copy3(out->normal, normal);
    out->geom = geom;
    out->body = m->geom_bodyid[geom];
  }
  return 1;
}

int mjf_overlapSphere(const mjModel* m, const mjData* d, const mjtNum center[3], mjtNum radius,
                      int bodyexclude, int mask, int* geoms, int maxgeoms) {
  if (!m || !d || !center || !isfinite(radius) || radius < 0 || maxgeoms < 0) return 0;
  for (int j = 0; j < 3; j++) if (!isfinite(center[j])) return 0;
  int found = 0;
  for (int g = 0; g < m->ngeom; g++) {
    if (m->geom_bodyid[g] == bodyexclude || !mj_geomInMask(m, g, mask)) continue;
    const mjtNum* pos = d->geom_xpos + 3*g;
    int overlaps;
    if (m->geom_type[g] == mjGEOM_PLANE) {
      const mjtNum* mat = d->geom_xmat + 9*g;
      mjtNum height = mat[2]*(center[0]-pos[0]) + mat[5]*(center[1]-pos[1]) +
                      mat[8]*(center[2]-pos[2]);
      overlaps = height <= radius;
    } else {
      mjtNum r = radius + m->geom_rbound[g];
      mjtNum delta[3];
      mju_sub3(delta, center, pos);
      overlaps = mju_dot3(delta, delta) <= r*r;
    }
    if (overlaps) {
      if (geoms && found < maxgeoms) geoms[found] = g;
      found++;
    }
  }
  return found;
}

// FEATHER training (integration16): the Filtered pair under its names before integration14, for
// feather's tests. The ray is feather's integration6 mjf_raycastFiltered, its result without the
// normal; the overlap is mjf_overlapSphere, which integration14 made of mjf_overlapSphereFiltered.
int mjf_raycastFiltered(const mjModel* m, const mjData* d,
                        const mjtNum origin[3], const mjtNum dir[3],
                        int bodyexclude, mjtBool include_static, int mask,
                        mjfRayHitNoNormal* out) {
  if (out) {
    out->dist = -1;
    mju_zero3(out->point);
    out->geom = out->body = -1;
  }
  if (!m || !d || !origin || !dir) return 0;
  for (int j = 0; j < 3; j++) if (!isfinite(origin[j]) || !isfinite(dir[j])) return 0;
  mjtNum v[3] = {dir[0], dir[1], dir[2]};
  if (mju_normalize3(v) <= mjMINVAL) return 0;
  int geom = -1;
  mjtNum dist = mj_rayMask(m, d, origin, v, NULL, include_static, bodyexclude, &geom, NULL, &mask);
  if (geom < 0 || dist < 0) return 0;
  if (out) {
    out->dist = dist;
    out->geom = geom;
    out->body = m->geom_bodyid[geom];
    mju_addScl3(out->point, origin, v, dist);
  }
  return 1;
}

int mjf_overlapSphereFiltered(const mjModel* m, const mjData* d,
                              const mjtNum center[3], mjtNum radius,
                              int bodyexclude, int mask, int* geoms, int maxgeoms) {
  return mjf_overlapSphere(m, d, center, radius, bodyexclude, mask, geoms, maxgeoms);
}

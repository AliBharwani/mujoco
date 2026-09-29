// Copyright 2026 Feather contributors. Licensed under Apache-2.0.
// Contact reuse across steps (mjf_setContactReuse): a geom pair whose relative pose has barely
// changed since its last collision replays that collision's contacts, their depths updated from
// points anchored on each geom, instead of running the collision function again.

#ifndef MUJOCO_SRC_ENGINE_ENGINE_COLLISION_REUSE_H_
#define MUJOCO_SRC_ENGINE_ENGINE_COLLISION_REUSE_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>

#ifdef __cplusplus
extern "C" {
#endif

// most contacts a pair may have to be replayed; pairs with more always collide
#define mjREUSEMAXCON 8

// d's cache when reuse is on, NULL otherwise
struct mjfContactCache_* mj_reuseCache(const mjData* d);

// start of a collision pass: advance the cache's step, drop entries of pairs no longer colliding
void mj_reuseBeginStep(struct mjfContactCache_* cache);

// replay the contacts of geom pair (g1, g2) into precon (room for maxcon): their count, or -1 when
// the pair must collide (no entry, moved past the tolerance, margin or size changed, not replayable)
int mj_reuseLookup(const mjModel* m, const mjData* d, struct mjfContactCache_* cache,
                   int g1, int g2, mjtNum margin, mjPreContact* precon, int maxcon);

// record what a collision of (g1, g2) with this margin returned
void mj_reuseStore(const mjModel* m, const mjData* d, struct mjfContactCache_* cache,
                   int g1, int g2, mjtNum margin, const mjPreContact* precon, int ncon);

// forget the pairs of heightfield `id`'s geoms whose footprint overlaps samples [rmin, rmax] x
// [cmin, cmax]; kinematics must be current
void mj_reuseHfieldEdit(const mjModel* m, const mjData* d, int id,
                        int rmin, int rmax, int cmin, int cmax);

// lifecycle, from engine_io.c: forget every entry (reset), make dest's cache a copy of src's (or
// free it when src has none), free d's cache
void mj_reuseClear(mjData* d);
void mj_reuseCopy(mjData* dest, const mjData* src);
void mj_reuseFree(mjData* d);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_COLLISION_REUSE_H_

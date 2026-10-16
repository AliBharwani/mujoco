// Copyright 2026 Feather contributors. Licensed under Apache-2.0.
// Contact reuse across steps. The gate is on the relative pose of the two geoms since the pair's
// last real collision: no point of g2 may have moved more than the tolerance in g1's frame
// (translation of g2's origin plus the rotation's chord at g2's bounding radius). Within it, each
// contact is replayed from a point anchored on each geom and a normal fixed to g1, its depth
// measured along that normal; a contact beyond the margin is dropped, as the collision function
// would have. Parameters (friction, solref, ...) are applied afterwards from the live model.

#include "engine/engine_collision_reuse.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

#include <mujoco/mjfeather.h>
#include <mujoco/mjmacro.h>
#include <mujoco/mujoco.h>
#include "engine/engine_collision_hfield.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_errmem.h"

typedef struct {
  int g1, g2;                          // key; g1 < 0: an empty slot
  int ncon;                            // contacts; -1: collide again (too many, or invalidated)
  int stamp;                           // the last pass that looked this pair up or stored it
  mjtNum margin;                       // margin + gap the collision used
  mjtNum size[6];                      // the two geoms' sizes then
  mjtNum rot[9];                       // g2's orientation in g1's frame then
  mjtNum pos[3];                       // g2's origin in g1's frame then
  mjtNum anchor1[3*mjREUSEMAXCON];     // contact points on g1, in g1's frame
  mjtNum anchor2[3*mjREUSEMAXCON];     // contact points on g2, in g2's frame
  mjtNum normal[3*mjREUSEMAXCON];      // normals, in g1's frame
  mjtNum tangent[3*mjREUSEMAXCON];     // first tangents, in g1's frame (zero: none)
} mjfPairEntry;

typedef struct mjfContactCache_ {
  mjtNum tolerance;         // most any point of g2 may move relative to g1 before a collision (m)
  int flags;                // mjfREUSE_EMPTY: also replay pairs that found no contact
  int capacity;             // slots, a power of two, kept at most half full
  int count;                // occupied slots
  int stamp;                // collision passes so far
  long long stats[4];       // lookups, replays, replays of no contact, collisions stored
  mjfPairEntry* slot;
  mjfPairEntry* spare;      // the same capacity, for dropping stale entries
} mjfContactCache;

#define mjREUSE_INITCAP 256


mjfContactCache* mj_reuseCache(const mjData* d) {
  return (mjfContactCache*)d->contactcache;
}


// pairs worth replaying: not two of plane, sphere and capsule (their functions are cheaper than a
// lookup), and no SDF (its contact bound exceeds what an entry holds)
static int replayable(const mjModel* m, int g1, int g2) {
  int t1 = m->geom_type[g1], t2 = m->geom_type[g2];
  if (t1 == mjGEOM_SDF || t2 == mjGEOM_SDF) {
    return 0;
  }
  return t1 > mjGEOM_CAPSULE || t2 > mjGEOM_CAPSULE || t1 == mjGEOM_HFIELD || t2 == mjGEOM_HFIELD;
}


static void clearSlots(mjfPairEntry* slot, int capacity) {
  for (int i=0; i < capacity; i++) {
    slot[i].g1 = -1;
  }
}


static inline int home(int capacity, int g1, int g2) {
  unsigned int h = (unsigned int)g1 * 2654435761u ^ (unsigned int)g2 * 2246822519u;
  return (int)(h & (unsigned int)(capacity - 1));
}


// the slot of (g1, g2), or of the empty slot where it would go
static mjfPairEntry* probe(mjfPairEntry* slot, int capacity, int g1, int g2) {
  for (int i=home(capacity, g1, g2); ; i = (i + 1) & (capacity - 1)) {
    mjfPairEntry* e = slot + i;
    if (e->g1 < 0 || (e->g1 == g1 && e->g2 == g2)) {
      return e;
    }
  }
}


static mjfContactCache* newCache(int capacity) {
  mjfContactCache* c = (mjfContactCache*)mju_malloc(sizeof(mjfContactCache));
  memset(c, 0, sizeof(mjfContactCache));
  c->capacity = capacity;
  c->slot = (mjfPairEntry*)mju_malloc(sizeof(mjfPairEntry) * capacity);
  c->spare = (mjfPairEntry*)mju_malloc(sizeof(mjfPairEntry) * capacity);
  clearSlots(c->slot, capacity);
  return c;
}


static void freeCache(mjfContactCache* c) {
  if (c) {
    mju_free(c->slot);
    mju_free(c->spare);
    mju_free(c);
  }
}


// rebuild without the entries no pass has touched since the previous one, doubling the capacity
// while the rest would fill more than a quarter of it
static void sweep(mjfContactCache* c) {
  int live = 0;
  for (int i=0; i < c->capacity; i++) {
    live += c->slot[i].g1 >= 0 && c->slot[i].stamp >= c->stamp - 1;
  }
  int capacity = c->capacity;
  while (4*(live + 1) > capacity) {
    capacity *= 2;
  }
  mjfPairEntry* dst = c->spare;
  if (capacity != c->capacity) {
    mju_free(c->spare);
    dst = (mjfPairEntry*)mju_malloc(sizeof(mjfPairEntry) * capacity);
  }
  clearSlots(dst, capacity);
  for (int i=0; i < c->capacity; i++) {
    const mjfPairEntry* e = c->slot + i;
    if (e->g1 >= 0 && e->stamp >= c->stamp - 1) {
      *probe(dst, capacity, e->g1, e->g2) = *e;
    }
  }
  if (capacity != c->capacity) {
    mju_free(c->slot);
    c->slot = dst;
    c->spare = (mjfPairEntry*)mju_malloc(sizeof(mjfPairEntry) * capacity);
    c->capacity = capacity;
  } else {
    c->spare = c->slot;
    c->slot = dst;
  }
  c->count = live;
}


void mj_reuseBeginStep(mjfContactCache* cache) {
  cache->stamp++;
}


// g2's pose in g1's frame: rot = R1' R2, pos = R1' (x2 - x1)
static void relativePose(const mjData* d, int g1, int g2, mjtNum rot[9], mjtNum pos[3]) {
  const mjtNum* R1 = d->geom_xmat + 9*g1;
  const mjtNum* R2 = d->geom_xmat + 9*g2;
  mjtNum dif[3];
  mju_sub3(dif, d->geom_xpos + 3*g2, d->geom_xpos + 3*g1);
  mju_mulMatTVec3(pos, R1, dif);
  for (int i=0; i < 3; i++) {
    for (int j=0; j < 3; j++) {
      rot[3*i+j] = R1[i]*R2[j] + R1[3+i]*R2[3+j] + R1[6+i]*R2[6+j];
    }
  }
}


int mj_reuseLookup(const mjModel* m, const mjData* d, mjfContactCache* c,
                   int g1, int g2, mjtNum margin, mjPreContact* precon, int maxcon) {
  if (!replayable(m, g1, g2)) {
    return -1;
  }
  c->stats[0]++;
  mjfPairEntry* e = probe(c->slot, c->capacity, g1, g2);
  if (e->g1 < 0) {
    return -1;
  }
  e->stamp = c->stamp;
  if (e->ncon < 0 || e->ncon > maxcon || e->margin != margin ||
      (!e->ncon && !(c->flags & mjfREUSE_EMPTY))) {
    return -1;
  }
  for (int k=0; k < 3; k++) {
    if (e->size[k] != m->geom_size[3*g1+k] || e->size[3+k] != m->geom_size[3*g2+k]) {
      return -1;
    }
  }

  // how far any point of g2 has moved in g1's frame: the origin's shift, plus the chord of the
  // rotation since (2 sin(angle/2), from its trace) at g2's bounding radius
  mjtNum rot[9], pos[3], shift[3];
  relativePose(d, g1, g2, rot, pos);
  mju_sub3(shift, pos, e->pos);
  mjtNum trace = 0;
  for (int i=0; i < 9; i++) {
    trace += e->rot[i] * rot[i];
  }
  mjtNum chord = 3 - trace > 0 ? mju_sqrt(3 - trace) : 0;
  if (mju_norm3(shift) + chord * m->geom_rbound[g2] > c->tolerance) {
    return -1;
  }

  const mjtNum* x1 = d->geom_xpos + 3*g1;
  const mjtNum* x2 = d->geom_xpos + 3*g2;
  const mjtNum* R1 = d->geom_xmat + 9*g1;
  const mjtNum* R2 = d->geom_xmat + 9*g2;
  int n = 0;
  for (int i=0; i < e->ncon; i++) {
    mjtNum p1[3], p2[3], normal[3], dif[3];
    mju_mulMatVec3(p1, R1, e->anchor1 + 3*i);
    mju_addTo3(p1, x1);
    mju_mulMatVec3(p2, R2, e->anchor2 + 3*i);
    mju_addTo3(p2, x2);
    mju_mulMatVec3(normal, R1, e->normal + 3*i);
    mju_sub3(dif, p2, p1);
    mjtNum dist = mju_dot3(dif, normal);
    if (dist > margin) {
      continue;
    }
    mjPreContact* con = precon + n++;
    con->dist = dist;
    mju_add3(con->pos, p1, p2);
    mju_scl3(con->pos, con->pos, 0.5);
    mju_copy3(con->normal, normal);
    mju_mulMatVec3(con->tangent, R1, e->tangent + 3*i);
  }
  c->stats[1]++;
  c->stats[2] += !e->ncon;
  return n;
}


void mj_reuseStore(const mjModel* m, const mjData* d, mjfContactCache* c,
                   int g1, int g2, mjtNum margin, const mjPreContact* precon, int ncon) {
  if (!replayable(m, g1, g2)) {
    return;
  }
  mjfPairEntry* e = probe(c->slot, c->capacity, g1, g2);
  if (e->g1 < 0) {
    if (2*(c->count + 1) > c->capacity) {
      sweep(c);
      e = probe(c->slot, c->capacity, g1, g2);
    }
    e->g1 = g1;
    e->g2 = g2;
    c->count++;
  }
  c->stats[3]++;
  e->stamp = c->stamp;
  e->margin = margin;
  mju_copy3(e->size, m->geom_size + 3*g1);
  mju_copy3(e->size + 3, m->geom_size + 3*g2);
  if (ncon > mjREUSEMAXCON) {
    e->ncon = -1;
    return;
  }
  e->ncon = ncon;
  relativePose(d, g1, g2, e->rot, e->pos);

  // the contact's points on each surface: its position -/+ half its distance along the normal
  const mjtNum* x1 = d->geom_xpos + 3*g1;
  const mjtNum* x2 = d->geom_xpos + 3*g2;
  const mjtNum* R1 = d->geom_xmat + 9*g1;
  const mjtNum* R2 = d->geom_xmat + 9*g2;
  for (int i=0; i < ncon; i++) {
    const mjPreContact* con = precon + i;
    mjtNum p[3];
    mju_addScl3(p, con->pos, con->normal, -0.5*con->dist);
    mju_subFrom3(p, x1);
    mju_mulMatTVec3(e->anchor1 + 3*i, R1, p);
    mju_addScl3(p, con->pos, con->normal, 0.5*con->dist);
    mju_subFrom3(p, x2);
    mju_mulMatTVec3(e->anchor2 + 3*i, R2, p);
    mju_mulMatTVec3(e->normal + 3*i, R1, con->normal);
    mju_mulMatTVec3(e->tangent + 3*i, R1, con->tangent);
  }
}


void mj_reuseHfieldEdit(const mjModel* m, const mjData* d, int id,
                        int rmin, int rmax, int cmin, int cmax) {
  mjfContactCache* c = mj_reuseCache(d);
  if (!c) {
    return;
  }
  int nr = m->hfield_nrow[id], nc = m->hfield_ncol[id];
  const mjtNum* size = m->hfield_size + 4*id;
  mjtNum dx = 2*size[0]/(nc-1), dy = 2*size[1]/(nr-1);
  for (int i=0; i < c->capacity; i++) {
    mjfPairEntry* e = c->slot + i;
    int hg = e->g1;
    if (hg < 0 || m->geom_type[hg] != mjGEOM_HFIELD || m->geom_dataid[hg] != id) {
      continue;
    }

    // the geom's footprint in samples, grown by the margin and the tolerance, and by two samples:
    // the collider's normals take central differences over the ring around the footprint
    mjtNum lo[3], hi[3];
    mjc_hfieldGeomBox(m, d, hg, e->g2, lo, hi);
    mjtNum pad = mju_max(0, e->margin) + c->tolerance;
    mjtNum x0 = (lo[0] - pad + size[0])/dx, x1 = (hi[0] + pad + size[0])/dx;
    mjtNum y0 = (lo[1] - pad + size[1])/dy, y1 = (hi[1] + pad + size[1])/dy;
    if (x1 + 2 < cmin || x0 - 2 > cmax || y1 + 2 < rmin || y0 - 2 > rmax) {
      continue;
    }
    e->ncon = -1;
  }
}


void mj_reuseClear(mjData* d) {
  mjfContactCache* c = mj_reuseCache(d);
  if (c) {
    clearSlots(c->slot, c->capacity);
    c->count = 0;
  }
}


void mj_reuseCopy(mjData* dest, const mjData* src) {
  const mjfContactCache* s = mj_reuseCache(src);
  mjfContactCache* c = mj_reuseCache(dest);
  if (!s) {
    freeCache(c);
    dest->contactcache = NULL;
    return;
  }
  if (!c || c->capacity != s->capacity) {
    freeCache(c);
    c = newCache(s->capacity);
    dest->contactcache = c;
  }
  mjfPairEntry* slot = c->slot;
  mjfPairEntry* spare = c->spare;
  *c = *s;
  c->slot = slot;
  c->spare = spare;
  for (int i=0; i < s->capacity; i++) {
    if (s->slot[i].g1 >= 0) {
      c->slot[i] = s->slot[i];
    } else {
      c->slot[i].g1 = -1;
    }
  }
}


void mj_reuseFree(mjData* d) {
  freeCache(mj_reuseCache(d));
  d->contactcache = NULL;
}


int mjf_setContactReuse(mjData* d, int enable, mjtNum tolerance, int flags) {
  if (!d || !(tolerance >= 0) || !isfinite(tolerance) || (flags & ~mjfREUSE_EMPTY)) {
    return -1;
  }
  if (!enable) {
    mj_reuseFree(d);
    return 0;
  }
  if (!d->contactcache) {
    d->contactcache = newCache(mjREUSE_INITCAP);
  }
  mjfContactCache* c = mj_reuseCache(d);
  mj_reuseClear(d);
  c->tolerance = tolerance;
  c->flags = flags;
  memset(c->stats, 0, sizeof(c->stats));
  return 0;
}


int mjf_contactReuseStats(const mjData* d, long long stats[4]) {
  const mjfContactCache* c = d ? mj_reuseCache(d) : NULL;
  if (!c || !stats) {
    return -1;
  }
  memcpy(stats, c->stats, sizeof(c->stats));
  return 0;
}

// Copyright 2026 Feather contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// Convex geoms against a heightfield's top surface. Upstream treats each cell as a solid prism,
// whose internal side faces give lateral normals even on a flat field. Over the grid interior this
// collider gives a smooth terrain normal and at most four contacts per pair, anchored at hull
// vertices, terrain samples and lattice crossings. mjc_SmoothHField is the collision table's entry:
// it leaves the side walls, the underside, unsupported geoms and fields set to upstream
// (hfield_collider, mjf_setHfieldCollider) to mjc_ConvexHField's prisms.

#include "engine/engine_collision_hfield.h"

#include <string.h>
#include <mujoco/mjmacro.h>
#include "engine/engine_collision_convex.h"
#include "engine/engine_collision_primitive.h"
#include "engine/engine_memory.h"
#include "engine/engine_ray.h"
#include "engine/engine_inline.h"
#include <math.h>
#include "engine/engine_util_blas.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_spatial.h"


//------------------------------ smooth heightfield contacts --------------------------------------
// The field is a height function, so a convex geom's clearance is its lower surface
// minus h(x, y). Heights stay the upstream triangulated surface (diagonal from sample
// (r, c) to (r+1, c+1)); normals come from a smooth field through the same samples:
// central-difference sample gradients, bilinear inside a cell. They are continuous
// across cells and exact on any plane, so flat or uniformly sloped terrain reduces to
// the plane functions. Contacts are anchored to geom and terrain features, never to clipped
// witnesses, and are at most four per pair.
//
// Polytopes (boxes, meshes): against the triangulated heights the clearance is minimal
// only at (1) lower vertices, (2) lattice samples under the lower surface, or (3)
// crossings of lower edges with the lattice's row, column and diagonal lines, so the
// deepest candidate is exact. Lattice features join the support polygon only where the
// terrain itself bends upward (a ridge line or a peak sample), a property of the terrain
// alone, so the choice cannot flip while the geom rocks; on a plane the contacts are
// mjc_PlaneBox's vertices.
//
// Spheres, capsules, ellipsoids, cylinders: the plane functions' points, each against the
// terrain below it on the smooth normals, plus the deepest terrain feature (a triangle's
// support point, a lattice sample, a sharp ridge edge) when it is deeper by more than a
// tenth of the margin. A capsule's side also rests on ridge lines and peak samples; like a
// hull edge it is straight, so those join its support along the axis.

static inline mjtNum hmin(mjtNum a, mjtNum b) { return a < b ? a : b; }
static inline mjtNum hmax(mjtNum a, mjtNum b) { return a > b ? a : b; }
static inline mjtNum hclip(mjtNum x, mjtNum lo, mjtNum hi) { return x < lo ? lo : x > hi ? hi : x; }
static inline int hfloor(mjtNum x) {
  int i = (int)x;
  return i - (x < i);
}
static inline mjtNum hdot3(const mjtNum a[3], const mjtNum b[3]) {
  return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}


//------------------------------ the terrain

typedef struct {
  const float* data;
  int nr, nc;
  mjtNum sx, sy, sz;  // half extents and height scale (hfield_size[0..2])
  mjtNum base;        // depth of the solid below z = 0 (hfield_size[3])
  mjtNum dx, dy;      // cell size
  mjtNum idx, idy;    // and its reciprocals
  mjtNum crease;      // smallest slope change that counts as a crease (above float rounding)
} HGrid;

static inline mjtNum sampleZ(const HGrid* g, int r, int c) {
  return g->data[(mjtSize)r*g->nc + c]*g->sz;
}

// Cell containing (x, y) with fractional coordinates in [0, 1].
static inline void gridCell(const HGrid* g, mjtNum x, mjtNum y, int* r, int* c,
                            mjtNum* u, mjtNum* v) {
  mjtNum U = (x + g->sx)*g->idx, V = (y + g->sy)*g->idy;
  int cc = hfloor(U), rr = hfloor(V);
  cc = mjMAX(0, mjMIN(g->nc - 2, cc));
  rr = mjMAX(0, mjMIN(g->nr - 2, rr));
  *c = cc;
  *r = rr;
  *u = hclip(U - cc, 0, 1);
  *v = hclip(V - rr, 0, 1);
}

// Height of the triangulated surface.
static mjtNum gridHeight(const HGrid* g, mjtNum x, mjtNum y) {
  int r, c;
  mjtNum u, v;
  gridCell(g, x, y, &r, &c, &u, &v);
  const float* p = g->data + (mjtSize)r*g->nc + c;
  mjtNum h00 = p[0], h10 = p[1], h01 = p[g->nc], h11 = p[g->nc + 1];
  mjtNum h = v >= u ? h00 + v*(h01 - h00) + u*(h11 - h01) :
                      h00 + u*(h10 - h00) + v*(h11 - h10);
  return h*g->sz;
}

// Central-difference gradient at a sample, one-sided on the border.
static inline void sampleGradient(const HGrid* g, int r, int c, mjtNum grad[2]) {
  int c0 = c > 0 ? c - 1 : c, c1 = c < g->nc - 1 ? c + 1 : c;
  int r0 = r > 0 ? r - 1 : r, r1 = r < g->nr - 1 ? r + 1 : r;
  const float* row = g->data + (mjtSize)r*g->nc;
  mjtNum sx = c1 - c0 == 2 ? 0.5*g->sz*g->idx : g->sz*g->idx;
  mjtNum sy = r1 - r0 == 2 ? 0.5*g->sz*g->idy : g->sz*g->idy;
  grad[0] = (row[c1] - row[c0])*sx;
  grad[1] = (g->data[(mjtSize)r1*g->nc + c] - g->data[(mjtSize)r0*g->nc + c])*sy;
}

// Unit normal of the surface z = h(x, y) with gradient (gx, gy).
static inline void slopeNormal(mjtNum gx, mjtNum gy, mjtNum normal[3]) {
  mjtNum scale = 1 / sqrt(1 + gx*gx + gy*gy);
  normal[0] = -gx*scale;
  normal[1] = -gy*scale;
  normal[2] = scale;
}

// Smooth unit normal: bilinear blend of the four sample gradients of the cell.
static void gridNormal(const HGrid* g, mjtNum x, mjtNum y, mjtNum normal[3]) {
  int r, c;
  mjtNum u, v, g00[2], g10[2], g01[2], g11[2];
  gridCell(g, x, y, &r, &c, &u, &v);
  sampleGradient(g, r, c, g00);
  sampleGradient(g, r, c + 1, g10);
  sampleGradient(g, r + 1, c, g01);
  sampleGradient(g, r + 1, c + 1, g11);
  mjtNum w00 = (1 - u)*(1 - v), w10 = u*(1 - v), w01 = (1 - u)*v, w11 = u*v;
  mjtNum gx = w00*g00[0] + w10*g10[0] + w01*g01[0] + w11*g11[0];
  mjtNum gy = w00*g00[1] + w10*g10[1] + w01*g01[1] + w11*g11[1];
  slopeNormal(gx, gy, normal);
}

// Unit normal of one of cell (r, c)'s triangles: upper (v >= u) or lower.
static void triangleNormal(const HGrid* g, int r, int c, int upper, mjtNum normal[3]) {
  mjtNum h00 = sampleZ(g, r, c), h10 = sampleZ(g, r, c + 1);
  mjtNum h01 = sampleZ(g, r + 1, c), h11 = sampleZ(g, r + 1, c + 1);
  mjtNum gx = upper ? (h11 - h01)*g->idx : (h10 - h00)*g->idx;
  mjtNum gy = upper ? (h01 - h00)*g->idy : (h11 - h10)*g->idy;
  slopeNormal(gx, gy, normal);
}


//------------------------------ candidates and their selection

typedef struct {
  mjtNum point[3];   // on the geom, hfield frame
  mjtNum normal[3];  // smooth terrain normal below it
  mjtNum dist;       // clearance to the tangent plane through the terrain point below
  mjtNum gap;        // vertical clearance
  int eligible;      // may join the support polygon
  int order;         // arrival order: ties go to the earlier candidate
} HCandidate;

// Candidates within margin. Those that may join the support polygon are kept in arrival
// order; of the rest only the deepest can be chosen, so only it is kept. Memory follows the
// eligible count, capped at HCANDIDATES; past the cap an eligible candidate competes only
// for the deepest, so the depth stays exact.
#define HCANDIDATES 1024

typedef struct {
  HCandidate* list;  // the eligible candidates, then one spare slot: the candidate being
                     // measured, and at the end the deepest of the rest
  int n, cap;
  HCandidate other;  // the deepest ineligible candidate (order < 0: none yet)
  int arrived;
  mjtNum margin;
  mjtNum prefilter;  // vertical gaps above this cannot be within margin (see clearsFootprint)
} HCandidates;

static void initCandidates(HCandidates* set, mjData* d, mjtNum estimate, mjtNum margin,
                           mjtNum prefilter) {
  set->cap = estimate < HCANDIDATES ? (int)estimate : HCANDIDATES;
  set->list = mjSTACKALLOC(d, set->cap + 1, HCandidate);
  set->n = 0;
  set->other.order = -1;
  set->arrived = 0;
  set->margin = margin;
  set->prefilter = prefilter;
}

// A geom surface point over the terrain, or NULL beyond the margin. The caller says whether
// it may join the support polygon, then keeps it.
static HCandidate* measureCandidate(const HGrid* g, HCandidates* set, const mjtNum point[3]) {
  mjtNum gap = point[2] - gridHeight(g, point[0], point[1]);
  if (gap > set->prefilter) {
    return NULL;
  }
  HCandidate* c = set->list + set->n;   // the spare slot
  gridNormal(g, point[0], point[1], c->normal);
  c->dist = gap*c->normal[2];
  if (c->dist > set->margin) {
    return NULL;
  }
  mji_copy3(c->point, point);
  c->gap = gap;
  c->eligible = 0;
  c->order = set->arrived++;
  return c;
}

static void keepCandidate(HCandidates* set, const HCandidate* c) {
  if (c->eligible && set->n < set->cap) {
    set->n++;   // measured in place
  } else if (set->other.order < 0 || c->dist < set->other.dist) {
    set->other = *c;
    set->other.eligible = 0;
  }
}

static void addCandidate(const HGrid* g, HCandidates* set, const mjtNum point[3], int eligible) {
  HCandidate* c = measureCandidate(g, set, point);
  if (c) {
    c->eligible = eligible;
    keepCandidate(set, c);
  }
}

// All kept candidates in one array: the eligible ones, then the deepest of the rest.
static int gatherCandidates(HCandidates* set) {
  if (set->other.order < 0) {
    return set->n;
  }
  set->list[set->n] = set->other;
  return set->n + 1;
}

static inline mjtNum cross2(const mjtNum* a, const mjtNum* b, const mjtNum* p) {
  return (b[0] - a[0])*(p[1] - a[1]) - (b[1] - a[1])*(p[0] - a[0]);
}

// Up to four contacts: the deepest, then the eligible candidates spanning the largest
// support polygon (farthest point, largest triangle, largest area added outside it).
// Ties go to the earlier candidate.
static int selectCandidates(HCandidates* set, int chosen[4], mjtNum tol) {
  const HCandidate* c = set->list;
  int n = gatherCandidates(set);
  if (!n) {
    return 0;
  }
  int best = 0;
  for (int i = 1; i < n; i++) {
    if (c[i].dist < c[best].dist || (c[i].dist == c[best].dist && c[i].order < c[best].order)) {
      best = i;
    }
  }
  chosen[0] = best;

  mjtNum far = tol*tol;
  int count = 1;
  for (int i = 0; i < n; i++) {
    if (!c[i].eligible) continue;
    mjtNum dx = c[i].point[0] - c[best].point[0], dy = c[i].point[1] - c[best].point[1];
    if (dx*dx + dy*dy > far) {
      far = dx*dx + dy*dy;
      chosen[1] = i;
      count = 2;
    }
  }
  if (count < 2) {
    return 1;
  }

  const mjtNum *a = c[chosen[0]].point, *b = c[chosen[1]].point;
  mjtNum area = tol*sqrt(far), sign = 0;
  for (int i = 0; i < n; i++) {
    if (!c[i].eligible) continue;
    mjtNum s = cross2(a, b, c[i].point);
    if (fabs(s) > area) {
      area = fabs(s);
      sign = s > 0 ? 1 : -1;
      chosen[2] = i;
      count = 3;
    }
  }
  if (count < 3) {
    return 2;
  }

  // with (a, b, p) counterclockwise, a point outside edge (i, j) has sign*cross2 < 0
  const mjtNum* p = c[chosen[2]].point;
  const mjtNum* tri[4] = {a, b, p, a};
  mjtNum outside = tol*sqrt(far);
  for (int i = 0; i < n; i++) {
    if (!c[i].eligible) continue;
    for (int e = 0; e < 3; e++) {
      mjtNum s = -sign*cross2(tri[e], tri[e + 1], c[i].point);
      if (s > outside) {
        outside = s;
        chosen[3] = i;
        count = 4;
      }
    }
  }
  return count;
}

// Write the chosen candidates as contacts, in the world frame.
static int emitContacts(const HCandidates* set, const int* chosen, int count,
                        mjPreContact* con, const mjtNum* hpos, const mjtNum* hmat) {
  for (int i = 0; i < count; i++) {
    const HCandidate* c = set->list + chosen[i];
    mjtNum point[3];
    mji_addScl3(point, c->point, c->normal, -0.5*c->dist);
    mji_mulMatVec3(con[i].pos, hmat, point);
    mji_addTo3(con[i].pos, hpos);
    mji_mulMatVec3(con[i].normal, hmat, c->normal);
    con[i].dist = c->dist;
    mji_zero3(con[i].tangent);
  }
  return count;
}


//------------------------------ shared setup

// Geom pose relative to the heightfield.
static void relativePose(const mjData* d, int hgeom, int g, mjtNum pos[3], mjtNum mat[9]) {
  mjtNum dif[3];
  mji_sub3(dif, d->geom_xpos + 3*g, d->geom_xpos + 3*hgeom);
  mji_mulMatTVec3(pos, d->geom_xmat + 9*hgeom, dif);
  mji_mulMatTMat3(mat, d->geom_xmat + 9*hgeom, d->geom_xmat + 9*g);
}

// The geom's bounding box in the heightfield frame, from its local AABB at pose (pos, mat).
static void geomBox(const mjtNum aabb[6], const mjtNum pos[3], const mjtNum mat[9],
                    mjtNum lo[3], mjtNum hi[3]) {
  mjtNum center[3];
  mji_mulMatVec3(center, mat, aabb);
  for (int k = 0; k < 3; k++) {
    mjtNum extent = fabs(mat[3*k])*aabb[3] + fabs(mat[3*k + 1])*aabb[4] +
                    fabs(mat[3*k + 2])*aabb[5];
    lo[k] = pos[k] + center[k] - extent;
    hi[k] = pos[k] + center[k] + extent;
  }
}

void mjc_hfieldGeomBox(const mjModel* m, const mjData* d, int hgeom, int g,
                       mjtNum lo[3], mjtNum hi[3]) {
  mjtNum pos[3], mat[9];
  relativePose(d, hgeom, g, pos, mat);
  geomBox(m->geom_aabb + 6*g, pos, mat, lo, hi);
}

// Footprint cell range of an xy box, clipped to the grid.
static void footprintCells(const HGrid* g, const mjtNum lo[2], const mjtNum hi[2],
                           int* rmin, int* rmax, int* cmin, int* cmax) {
  *cmin = mjMAX(0, hfloor((lo[0] + g->sx)*g->idx));
  *cmax = mjMIN(g->nc - 2, hfloor((hi[0] + g->sx)*g->idx));
  *rmin = mjMAX(0, hfloor((lo[1] + g->sy)*g->idy));
  *rmax = mjMIN(g->nr - 2, hfloor((hi[1] + g->sy)*g->idy));
}

// The geom must lie over the grid interior and above the base; otherwise the solid
// prisms' side and bottom walls apply and the caller falls back to them.
static int insideGrid(const HGrid* g, const mjtNum lo[3], const mjtNum hi[3], mjtNum margin) {
  return lo[0] > -g->sx + margin && hi[0] < g->sx - margin &&
         lo[1] > -g->sy + margin && hi[1] < g->sy - margin && lo[2] > -g->base + margin;
}

// Height range over the footprint's samples and the ring around them (their central
// differences give the normals). It bounds the smooth normal's tilt, so a candidate
// can be within margin only if its vertical gap is below margin/nz_min: the prefilter.
// Returns 1 when the geom's lowest point clears the highest sample by that much.
static int clearsFootprint(const HGrid* g, int rmin, int rmax, int cmin, int cmax,
                           mjtNum zmin, mjtNum margin, mjtNum* high, mjtNum* prefilter) {
  mjtNum low = mjMAXVAL;
  *high = -mjMAXVAL;
  for (int r = mjMAX(0, rmin - 1); r <= mjMIN(g->nr - 1, rmax + 2); r++) {
    const float* row = g->data + (mjtSize)r*g->nc;
    for (int c = mjMAX(0, cmin - 1); c <= mjMIN(g->nc - 1, cmax + 2); c++) {
      mjtNum z = row[c]*g->sz;
      low = hmin(low, z);
      *high = hmax(*high, z);
    }
  }
  mjtNum range = *high - low;
  mjtNum tilt = range*range*(g->idx*g->idx + g->idy*g->idy);
  *prefilter = margin*sqrt(1 + tilt)*(1 + 1e-9) + 1e-12*(1 + fabs(*high));
  return zmin - *high > *prefilter;
}


//------------------------------ polytopes

// A non-vertical face as the height function z = e - a*x - b*y.
typedef struct {
  mjtNum a, b, e;
} HPlane;

static const int boxface[6][4] = {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4},
                                  {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
static const int boxvertface[8][3] = {{0, 2, 4}, {1, 2, 4}, {0, 3, 4}, {1, 3, 4},
                                      {0, 2, 5}, {1, 2, 5}, {0, 3, 5}, {1, 3, 5}};

// A sample that some plane can rest on alone: its six neighbours all lie below one plane
// through it. Along each lattice line through the sample, that plane's rise per cell must
// exceed the neighbour ahead and stay under the drop to the one behind; the column and
// row rises add up to the diagonal's.
static int samplePeak(const HGrid* g, int r, int c) {
  static const int step[3][2] = {{0, 1}, {1, 0}, {1, 1}};  // column (x), row (y), diagonal
  mjtNum h = sampleZ(g, r, c), lo[3], hi[3];
  mjtNum tol = g->crease*hmin(g->dx, g->dy);
  for (int k = 0; k < 3; k++) {
    int ra = r + step[k][0], ca = c + step[k][1], rb = r - step[k][0], cb = c - step[k][1];
    lo[k] = ra < g->nr && ca < g->nc ? sampleZ(g, ra, ca) - h : -mjMAXVAL;
    hi[k] = rb >= 0 && cb >= 0 ? h - sampleZ(g, rb, cb) : mjMAXVAL;
    if (lo[k] + tol >= hi[k]) {
      return 0;
    }
  }
  return lo[0] + lo[1] + tol < hi[2] && lo[2] + tol < hi[0] + hi[1];
}

// Box or mesh hull (a box is 8 vertices and 6 quads).
typedef struct {
  const mjModel* m;
  int mesh, nv, npoly;
  const float* vert;        // mesh vertices, local frame (NULL for a box)
  const mjtNum* normal;     // mesh polygon normals, local frame
  const mjtNum* size;       // box half sizes
} HHull;

static inline void hullVertex(const HHull* h, int i, mjtNum v[3]) {
  for (int j = 0; j < 3; j++) {
    v[j] = h->vert ? h->vert[3*i + j] : ((i & (1 << j)) ? h->size[j] : -h->size[j]);
  }
}

static inline const int* hullPolygon(const HHull* h, int f, int* count) {
  if (!h->vert) {
    *count = 4;
    return boxface[f];
  }
  int poly = h->m->mesh_polyadr[h->mesh] + f;
  *count = h->m->mesh_polyvertnum[poly];
  return h->m->mesh_polyvert + h->m->mesh_polyvertadr[poly];
}

// Polygons around vertex i (mesh polygon ids are local to the mesh).
static inline const int* hullVertexPolygons(const HHull* h, int i, int* count) {
  if (!h->vert) {
    *count = 3;
    return boxvertface[i];
  }
  int v = h->m->mesh_vertadr[h->mesh] + i;
  *count = h->m->mesh_polymapnum[v];
  return h->m->mesh_polymap + h->m->mesh_polymapadr[v];
}

static inline void hullNormal(const HHull* h, int f, mjtNum n[3]) {
  if (!h->vert) {
    n[0] = n[1] = n[2] = 0;
    n[f/2] = f % 2 ? 1 : -1;
  } else {
    mji_copy3(n, h->normal + 3*f);
  }
}

// Vertical gap at parameter t along the edge a-b.
static inline mjtNum edgeGap(const HGrid* g, const mjtNum* a, const mjtNum* b, mjtNum t) {
  t = hclip(t, 0, 1);
  return a[2] + t*(b[2] - a[2]) - gridHeight(g, a[0] + t*(b[0] - a[0]), a[1] + t*(b[1] - a[1]));
}

// Candidates where the lower edge a-b crosses the lattice's column, row and diagonal
// lines (U = k, V = k, U - V = k in cell units). A crossing joins the support polygon
// only where the line is a ridge: the terrain's slope along the edge drops across it.
// The edge is straight, so that is the clearance's second difference, and it does not
// depend on how the geom is tilted. A crossing higher than `top` (the highest footprint sample
// plus the prefilter, and a rounding guard) is beyond the margin, so the terrain is not sampled.
static void addEdgeCrossings(const HGrid* g, HCandidates* set, const mjtNum* a,
                             const mjtNum* b, mjtNum top) {
  mjtNum len = -1;  // measured on first use
  mjtNum Ua = (a[0] + g->sx)*g->idx, Va = (a[1] + g->sy)*g->idy;
  mjtNum Ub = (b[0] + g->sx)*g->idx, Vb = (b[1] + g->sy)*g->idy;
  mjtNum fa[3] = {Ua, Va, Ua - Va}, fb[3] = {Ub, Vb, Ub - Vb};
  for (int f = 0; f < 3; f++) {
    mjtNum lo = hmin(fa[f], fb[f]), hi = hmax(fa[f], fb[f]);
    if (hi - lo < mjMINVAL) {
      continue;
    }
    mjtNum eps = 1e-4 / (hi - lo);  // a ten-thousandth of a cell along the edge
    for (int k = hfloor(lo) + 1; k < hi; k++) {
      mjtNum t = (k - fa[f]) / (fb[f] - fa[f]);
      mjtNum point[3] = {a[0] + t*(b[0] - a[0]), a[1] + t*(b[1] - a[1]), a[2] + t*(b[2] - a[2])};
      if (point[2] > top) continue;
      HCandidate* c = measureCandidate(g, set, point);
      if (c) {
        // at an end the crossing is the vertex, already a candidate
        if (t > eps && t < 1 - eps) {
          if (len < 0) {
            len = sqrt((b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1]));
          }
          mjtNum bend = edgeGap(g, a, b, t - eps) + edgeGap(g, a, b, t + eps) - 2*c->gap;
          c->eligible = bend > g->crease*eps*len;
        }
        keepCandidate(set, c);
      }
    }
  }
}

// Boxes and meshes with compiled polygons. Only the hull near the terrain is expanded:
// every candidate feature lies on a lower face with a vertex at most `top` high (the
// highest footprint sample plus the prefilter), so the lower polygons around those
// vertices carry them all.
static int smoothPolytope(const mjModel* m, mjData* d, mjPreContact* con, int hgeom, int g,
                          mjtNum margin, const HGrid* grid) {
  int type = m->geom_type[g], mesh = m->geom_dataid[g];
  if (type == mjGEOM_MESH && (mesh < 0 || m->mesh_polynum[mesh] <= 0)) {
    return -1;
  }
  mjtNum pos[3], mat[9];
  relativePose(d, hgeom, g, pos, mat);
  HHull hull = {m, mesh, type == mjGEOM_BOX ? 8 : m->mesh_vertnum[mesh],
                type == mjGEOM_BOX ? 6 : m->mesh_polynum[mesh],
                type == mjGEOM_MESH ? m->mesh_vert + 3*m->mesh_vertadr[mesh] : NULL,
                type == mjGEOM_MESH ? m->mesh_polynormal + 3*m->mesh_polyadr[mesh] : NULL,
                m->geom_size + 3*g};
  int nv = hull.nv, npoly = hull.npoly;
  mjtNum scale = m->geom_rbound[g] > 0 ? m->geom_rbound[g] : 1;

  // footprint from the geom's bounding box
  mjtNum lo[3], hi[3];
  geomBox(m->geom_aabb + 6*g, pos, mat, lo, hi);
  if (!insideGrid(grid, lo, hi, margin)) {
    return -1;
  }

  mj_markStack(d);
  mjtNum* vert = mj_stackAllocNum(d, 3*nv);
  int* local = mj_stackAllocInt(d, nv);         // index among the expanded vertices, or -1
  int* key = mj_stackAllocInt(d, 3*nv);         // lattice column, row and diagonal band
  unsigned char* mark = mjSTACKALLOC(d, npoly, unsigned char);  // 0 new, 1 not lower, 2 lower
  memset(mark, 0, npoly);

  // heights of every vertex (one row of the rotation)
  mjtNum zmin = mjMAXVAL;
  for (int i = 0; i < nv; i++) {
    mjtNum v[3];
    hullVertex(&hull, i, v);
    vert[3*i + 2] = mat[6]*v[0] + mat[7]*v[1] + mat[8]*v[2] + pos[2];
    zmin = hmin(zmin, vert[3*i + 2]);
    local[i] = -1;
  }
  int rmin, rmax, cmin, cmax;
  mjtNum high, prefilter;
  footprintCells(grid, lo, hi, &rmin, &rmax, &cmin, &cmax);
  if (clearsFootprint(grid, rmin, rmax, cmin, cmax, zmin, margin, &high, &prefilter)) {
    mj_freeStack(d);
    return 0;
  }
  mjtNum top = high + prefilter;

  // lower polygons around the band's vertices, and their vertices
  int* face = mj_stackAllocInt(d, npoly);
  int* expanded = mj_stackAllocInt(d, nv);
  int nface = 0, nexpanded = 0, nside = 0;
  for (int i = 0; i < nv; i++) {
    if (vert[3*i + 2] > top) continue;
    int count;
    const int* polys = hullVertexPolygons(&hull, i, &count);
    for (int k = 0; k < count; k++) {
      int f = polys[k];
      if (mark[f]) continue;
      mjtNum n[3];
      hullNormal(&hull, f, n);
      mark[f] = mat[6]*n[0] + mat[7]*n[1] + mat[8]*n[2] < -1e-12 ? 2 : 1;
      if (mark[f] != 2) continue;
      face[nface++] = f;
      int nfv;
      const int* ids = hullPolygon(&hull, f, &nfv);
      nside += nfv;
      for (int j = 0; j < nfv; j++) {
        int id = ids[j];
        if (local[id] >= 0) continue;
        local[id] = nexpanded;
        expanded[nexpanded++] = id;
        mjtNum v[3], z = vert[3*id + 2];
        hullVertex(&hull, id, v);
        mji_mulMatVec3(vert + 3*id, mat, v);
        mji_addTo3(vert + 3*id, pos);
        vert[3*id + 2] = z;  // keep the height pass's value bit for bit
        mjtNum u = (vert[3*id] + grid->sx)*grid->idx, w = (vert[3*id + 1] + grid->sy)*grid->idy;
        key[3*id] = hfloor(u);
        key[3*id + 1] = hfloor(w);
        key[3*id + 2] = hfloor(u - w);
      }
    }
  }

  // eligible candidates: at most the expanded vertices, the footprint samples, and the
  // crossings of the band's edges, each spanning at most the footprint box
  mjtNum wu = (hi[0] - lo[0])*grid->idx, wv = (hi[1] - lo[1])*grid->idy;
  mjtNum estimate = nexpanded + (mjtNum)(rmax - rmin + 2)*(cmax - cmin + 2) +
                    (mjtNum)nside*(2*(wu + wv) + 3);
  HCandidates set;
  initCandidates(&set, d, estimate, margin, prefilter);

  // (1) lower vertices in the band
  for (int k = 0; k < nexpanded; k++) {
    const mjtNum* v = vert + 3*expanded[k];
    if (v[2] <= top) {
      addCandidate(grid, &set, v, 1);
    }
  }

  // (2) samples under the lower surface. Lower-face planes support the hull, so at a
  // sample under the geom the highest of them is the surface itself. Only the band's
  // faces are at hand: the highest of those is exact when its polygon contains the
  // sample; otherwise the face above lies outside the band, higher than `top`. A sample
  // joins the support polygon when it is a peak of the terrain (samplePeak).
  HPlane* plane = NULL;
  for (int r = rmin; r <= rmax + 1; r++) {
    for (int c = cmin; c <= cmax + 1; c++) {
      mjtNum h = sampleZ(grid, r, c);
      if (zmin - h > prefilter) continue;
      if (!plane) {
        plane = mjSTACKALLOC(d, nface + 1, HPlane);
        for (int k = 0; k < nface; k++) {
          mjtNum n[3], w[3];
          int count;
          const int* ids = hullPolygon(&hull, face[k], &count);
          hullNormal(&hull, face[k], w);
          mji_mulMatVec3(n, mat, w);
          mjtNum inv = 1/n[2];
          plane[k].a = n[0]*inv;
          plane[k].b = n[1]*inv;
          plane[k].e = hdot3(n, vert + 3*ids[0])*inv;
        }
      }
      mjtNum x = c*grid->dx - grid->sx, y = r*grid->dy - grid->sy;
      mjtNum z = -mjMAXVAL, limit = h + prefilter;
      int best = -1;
      for (int k = 0; k < nface; k++) {
        mjtNum zk = plane[k].e - plane[k].a*x - plane[k].b*y;
        if (zk > z) {
          z = zk;
          best = k;
          if (z > limit) break;
        }
      }
      if (best < 0 || z > limit) continue;
      int count;
      const int* ids = hullPolygon(&hull, face[best], &count);
      int pos_turn = 0, neg_turn = 0;
      for (int j = 0, prev = ids[count - 1]; j < count; prev = ids[j++]) {
        const mjtNum *a = vert + 3*prev, *b = vert + 3*ids[j];
        mjtNum turn = (b[0] - a[0])*(y - a[1]) - (b[1] - a[1])*(x - a[0]);
        pos_turn |= turn > 1e-12*scale*scale;
        neg_turn |= turn < -1e-12*scale*scale;
      }
      if (pos_turn && neg_turn) continue;
      mjtNum point[3] = {x, y, z};
      HCandidate* cand = measureCandidate(grid, &set, point);
      if (cand) {
        cand->eligible = samplePeak(grid, r, c);
        keepCandidate(&set, cand);
      }
    }
  }

  // (3) the band's lower edges across the lattice lines, once each (an open-addressing set
  // of expanded-vertex pairs, one slot per polygon side: lower faces share most edges, so it is
  // usually about half full, and it can never overflow)
  int slots = 16;
  while (slots < nside) slots *= 2;
  unsigned long long* seen = mjSTACKALLOC(d, slots, unsigned long long);
  memset(seen, 0, sizeof(unsigned long long)*slots);
  for (int k = 0; k < nface; k++) {
    int count;
    const int* ids = hullPolygon(&hull, face[k], &count);
    for (int j = 0, i0 = ids[count - 1]; j < count; i0 = ids[j++]) {
      int i1 = ids[j];
      // an edge within one cell of every line family crosses no line
      if (key[3*i0] == key[3*i1] && key[3*i0 + 1] == key[3*i1 + 1] &&
          key[3*i0 + 2] == key[3*i1 + 2]) continue;
      const mjtNum *a = vert + 3*i0, *b = vert + 3*i1;
      if (hmin(a[2], b[2]) > top) continue;
      unsigned long long pair = (unsigned long long)mjMIN(local[i0], local[i1])*nexpanded +
                                mjMAX(local[i0], local[i1]) + 1;
      int slot = (int)((pair*0x9E3779B97F4A7C15ULL) >> 40) & (slots - 1);
      while (seen[slot] && seen[slot] != pair) slot = (slot + 1) & (slots - 1);
      if (seen[slot]) continue;
      seen[slot] = pair;
      addEdgeCrossings(grid, &set, a, b, top + 1e-9*(1 + fabs(top)));
    }
  }

  int chosen[4];
  int ncon = selectCandidates(&set, chosen, 1e-6*scale);
  ncon = emitContacts(&set, chosen, ncon, con, d->geom_xpos + 3*hgeom, d->geom_xmat + 9*hgeom);
  mj_freeStack(d);
  return ncon;
}


//------------------------------ spheres, capsules, ellipsoids and cylinders

// A curved geom is upstream's CCD object with its pose in the heightfield frame, so its support
// point is upstream's support function.

// Height of the geom's lowest surface point over (x, y), or mjMAXVAL off it.
static mjtNum shapeBottom(const mjCCDObj* s, mjtNum x, mjtNum y) {
  mjtNum start[3] = {x, y, s->pos[2] - 2*(s->size[0] + s->size[1] + s->size[2]) - 1};
  mjtNum up[3] = {0, 0, 1};
  mjtNum t = mju_rayGeom(s->pos, s->mat, s->size, start, up, s->geom_type, NULL);
  return t >= 0 ? start[2] + t : mjMAXVAL;
}

// The geom's point whose outward normal opposes the terrain normal below it, by
// fixed-point iteration on the smooth normal field (on a plane the first step is exact).
static void supportOnTerrain(const HGrid* g, mjCCDObj* s, mjtNum point[3], mjtNum normal[3]) {
  mjtNum dir[3];
  gridNormal(g, s->pos[0], s->pos[1], normal);
  for (int k = 0; k < 3; k++) {
    mji_scl3(dir, normal, -1);
    s->support(point, s, dir);
    gridNormal(g, point[0], point[1], normal);
  }
  mji_scl3(dir, normal, -1);
  s->support(point, s, dir);
}

// The lowest clearance along the lattice edge from sample (r0, c0) to (r1, c1): the
// geom's lower surface is convex, so the gap is unimodal along the edge. Nine samples over
// the part of the edge under the geom's box (lo, hi), then a golden-section search around
// the best one.
static inline mjtNum edgeClearance(const mjCCDObj* s, const mjtNum p0[3], const mjtNum p1[3],
                                   mjtNum t) {
  return shapeBottom(s, p0[0] + t*(p1[0] - p0[0]), p0[1] + t*(p1[1] - p0[1])) -
         (p0[2] + t*(p1[2] - p0[2]));
}

static void addEdgeMinimum(const HGrid* g, HCandidates* set, const mjCCDObj* s,
                           int r0, int c0, int r1, int c1, const mjtNum lo[2],
                           const mjtNum hi[2]) {
  mjtNum x0 = c0*g->dx - g->sx, y0 = r0*g->dy - g->sy, z0 = sampleZ(g, r0, c0);
  mjtNum x1 = c1*g->dx - g->sx, y1 = r1*g->dy - g->sy, z1 = sampleZ(g, r1, c1);
  mjtNum e0[3] = {x0, y0, z0}, e1[3] = {x1, y1, z1};
  mjtNum ta = 0, tb = 1, p0[2] = {x0, y0}, dp[2] = {x1 - x0, y1 - y0};
  for (int k = 0; k < 2; k++) {
    if (fabs(dp[k]) < mjMINVAL) {
      if (p0[k] < lo[k] || p0[k] > hi[k]) return;
      continue;
    }
    mjtNum u = (lo[k] - p0[k])/dp[k], v = (hi[k] - p0[k])/dp[k];
    ta = hmax(ta, hmin(u, v));
    tb = hmin(tb, hmax(u, v));
  }
  if (ta > tb) {
    return;
  }
  mjtNum span = tb - ta, best = mjMAXVAL, tbest = ta;
  for (int i = 0; i <= 8; i++) {
    mjtNum t = ta + span*i/8.0;
    mjtNum f = edgeClearance(s, e0, e1, t);
    if (f < best) {
      best = f;
      tbest = t;
    }
  }
  if (best >= mjMAXVAL/2) {
    return;
  }
  const mjtNum phi = 0.6180339887498949;
  mjtNum a = hmax(ta, tbest - 0.125*span), b = hmin(tb, tbest + 0.125*span);
  mjtNum t1 = b - phi*(b - a), t2 = a + phi*(b - a);
  mjtNum f1 = edgeClearance(s, e0, e1, t1), f2 = edgeClearance(s, e0, e1, t2);
  for (int k = 0; k < 24; k++) {
    if (f1 < f2) {
      b = t2; t2 = t1; f2 = f1;
      t1 = b - phi*(b - a);
      f1 = edgeClearance(s, e0, e1, t1);
    } else {
      a = t1; t1 = t2; f1 = f2;
      t2 = a + phi*(b - a);
      f2 = edgeClearance(s, e0, e1, t2);
    }
  }
  mjtNum t = f1 < f2 ? t1 : t2;
  if (hmin(f1, f2) > best) {
    t = tbest;
  }
  mjtNum x = x0 + t*(x1 - x0), y = y0 + t*(y1 - y0);
  mjtNum point[3] = {x, y, shapeBottom(s, x, y)};
  if (point[2] < mjMAXVAL) {
    addCandidate(g, set, point, 0);
  }
}

// Candidates where a capsule's axis, projected on the terrain, crosses the lattice lines.
// In the vertical plane through the axis the side's lowest line is straight, so between
// the caps its clearance is minimal only at those crossings that are ridges (or at a
// peak sample under the side); a ridge crossing joins the support.
static void addAxisCrossings(const HGrid* g, HCandidates* set, const mjCCDObj* s) {
  mjtNum a[2] = {s->pos[0] + s->mat[2]*s->size[1], s->pos[1] + s->mat[5]*s->size[1]};
  mjtNum b[2] = {s->pos[0] - s->mat[2]*s->size[1], s->pos[1] - s->mat[5]*s->size[1]};
  mjtNum len = sqrt((b[0] - a[0])*(b[0] - a[0]) + (b[1] - a[1])*(b[1] - a[1]));
  if (len < mjMINVAL) {
    return;
  }
  mjtNum Ua = (a[0] + g->sx)*g->idx, Va = (a[1] + g->sy)*g->idy;
  mjtNum Ub = (b[0] + g->sx)*g->idx, Vb = (b[1] + g->sy)*g->idy;
  mjtNum fa[3] = {Ua, Va, Ua - Va}, fb[3] = {Ub, Vb, Ub - Vb};
  for (int f = 0; f < 3; f++) {
    mjtNum lo = hmin(fa[f], fb[f]), hi = hmax(fa[f], fb[f]);
    if (hi - lo < mjMINVAL) {
      continue;
    }
    mjtNum eps = 1e-4 / (hi - lo);
    for (int k = hfloor(lo) + 1; k < hi; k++) {
      mjtNum t = (k - fa[f]) / (fb[f] - fa[f]);
      mjtNum x = a[0] + t*(b[0] - a[0]), y = a[1] + t*(b[1] - a[1]);
      mjtNum point[3] = {x, y, shapeBottom(s, x, y)};
      if (point[2] >= mjMAXVAL) continue;
      HCandidate* c = measureCandidate(g, set, point);
      if (c) {
        mjtNum ex = eps*(b[0] - a[0]), ey = eps*(b[1] - a[1]);
        mjtNum bend = 2*gridHeight(g, x, y) - gridHeight(g, x - ex, y - ey) -
                      gridHeight(g, x + ex, y + ey);
        c->eligible = bend > g->crease*eps*len;
        keepCandidate(set, c);
      }
    }
  }
}

// A capsule's support lies along its axis: the eligible candidates (the end points, ridge
// crossings, peak samples) extreme along it, the deepest eligible one between them, and
// the deepest other feature when it beats them all by the slack.
static int selectCapsule(HCandidates* set, const mjtNum axis[3], mjtNum slack,
                         int chosen[4]) {
  const HCandidate* c = set->list;
  int n = gatherCandidates(set), lo = -1, hi = -1;
  mjtNum plo = 0, phi = 0, deepest = mjMAXVAL;
  for (int i = 0; i < n; i++) {
    if (!c[i].eligible) continue;
    mjtNum p = hdot3(c[i].point, axis);
    if (lo < 0 || p < plo) {
      lo = i;
      plo = p;
    }
    if (hi < 0 || p > phi) {
      hi = i;
      phi = p;
    }
    deepest = hmin(deepest, c[i].dist);
  }
  int between = -1, feature = -1;
  for (int i = 0; i < n; i++) {
    if (c[i].eligible) {
      if (i != lo && i != hi && (between < 0 || c[i].dist < c[between].dist)) {
        between = i;
      }
    } else if (c[i].dist < deepest - slack &&
               (feature < 0 || c[i].dist < c[feature].dist)) {
      feature = i;
    }
  }
  int count = 0;
  if (lo >= 0) chosen[count++] = lo;
  if (hi >= 0 && hi != lo) chosen[count++] = hi;
  if (between >= 0) chosen[count++] = between;
  if (feature >= 0) chosen[count++] = feature;
  return count;
}

static int smoothCurved(const mjModel* m, mjData* d, mjPreContact* con, int hgeom, int g,
                        mjtNum margin, const HGrid* grid) {
  mjCCDObj shape;
  mjc_initCCDObj(&shape, m, d, g, 0);
  relativePose(d, hgeom, g, shape.pos, shape.mat);

  // axis-aligned bounds from the support function
  mjtNum lo[3], hi[3];
  for (int j = 0; j < 3; j++) {
    mjtNum dir[3] = {0, 0, 0}, res[3];
    dir[j] = 1;
    shape.support(res, &shape, dir);
    hi[j] = res[j];
    dir[j] = -1;
    shape.support(res, &shape, dir);
    lo[j] = res[j];
  }
  if (!insideGrid(grid, lo, hi, margin)) {
    return -1;
  }
  int rmin, rmax, cmin, cmax;
  mjtNum high, prefilter;
  footprintCells(grid, lo, hi, &rmin, &rmax, &cmin, &cmax);
  if (clearsFootprint(grid, rmin, rmax, cmin, cmax, lo[2], margin, &high, &prefilter)) {
    return 0;
  }
  mjtNum scale = m->geom_rbound[g] > 0 ? m->geom_rbound[g] : 1;
  mjtNum slack = 0.1*margin + 1e-6*scale;  // a feature must beat the plane points by this

  mjtNum flattest = scale;
  if (shape.geom_type == mjGEOM_ELLIPSOID) {
    mjtNum longest = hmax(shape.size[0], hmax(shape.size[1], shape.size[2]));
    mjtNum shortest = hmin(shape.size[0], hmin(shape.size[1], shape.size[2]));
    flattest = hmax(scale, longest*longest/hmax(shortest, mjMINVAL));
  }

  // the plane functions' points, each measured against the terrain below it
  mjtNum axis[3] = {shape.mat[2], shape.mat[5], shape.mat[8]};
  mjtNum points[4][3], normal[3];
  int nplane = 0;
  if (shape.geom_type == mjGEOM_CAPSULE) {
    // mjc_PlaneCapsule: the two end spheres (a capsule of zero length is its end sphere)
    mjCCDObj ball = shape;
    ball.size[1] = 0;
    for (int e = 0; e < 2; e++) {
      mji_addScl3(ball.pos, shape.pos, axis, e ? -shape.size[1] : shape.size[1]);
      supportOnTerrain(grid, &ball, points[nplane++], normal);
    }
  } else if (shape.geom_type == mjGEOM_CYLINDER) {
    // mjc_PlaneCylinder's four points, under the terrain normal at its deepest one
    mjtNum ax[3], vec[3], vec1[3];
    supportOnTerrain(grid, &shape, points[0], normal);
    mjc_cylinderRim(normal, shape.mat, shape.size, ax, vec, vec1);
    mji_add3(points[0], shape.pos, vec);
    mji_addTo3(points[0], ax);
    mji_add3(points[1], shape.pos, vec);
    mji_subFrom3(points[1], ax);
    for (int k = 0; k < 2; k++) {
      mji_addScl3(points[2 + k], shape.pos, vec1, k ? -1 : 1);
      mji_addTo3(points[2 + k], ax);
      mji_addToScl3(points[2 + k], vec, -0.5);
    }
    nplane = 4;
  } else {
    // mjc_PlaneSphere, and mjc_PlaneConvex for ellipsoids: the support point
    supportOnTerrain(grid, &shape, points[nplane++], normal);
  }

  mj_markStack(d);
  // eligible candidates: the plane points, and for a capsule the samples under it and its
  // axis crossings
  mjtNum estimate = nplane;
  if (shape.geom_type == mjGEOM_CAPSULE) {
    estimate += (mjtNum)(rmax - rmin + 2)*(cmax - cmin + 2) + 3*(rmax - rmin + cmax - cmin + 4);
  }
  HCandidates set;
  initCandidates(&set, d, estimate, margin, prefilter);
  for (int i = 0; i < nplane; i++) {
    addCandidate(grid, &set, points[i], 1);
  }
  int nshape = set.n;

  // terrain features: (1) each triangle's support point inside it
  for (int r = rmin; r <= rmax; r++) {
    for (int c = cmin; c <= cmax; c++) {
      mjtNum x = c*grid->dx - grid->sx, y = r*grid->dy - grid->sy;
      for (int upper = 0; upper < 2; upper++) {
        mjtNum n[3], dir[3], s[3];
        triangleNormal(grid, r, c, upper, n);
        mji_scl3(dir, n, -1);
        shape.support(s, &shape, dir);
        mjtNum u = (s[0] - x)*grid->idx, v = (s[1] - y)*grid->idy;
        if (u < 0 || v < 0 || u > 1 || v > 1 || (upper ? v < u : v > u)) continue;
        addCandidate(grid, &set, s, 0);
      }
    }
  }
  // (2) samples under the geom
  for (int r = rmin; r <= rmax + 1; r++) {
    for (int c = cmin; c <= cmax + 1; c++) {
      mjtNum h = sampleZ(grid, r, c);
      if (lo[2] - h > prefilter) continue;
      mjtNum x = c*grid->dx - grid->sx, y = r*grid->dy - grid->sy;
      mjtNum point[3] = {x, y, shapeBottom(&shape, x, y)};
      if (point[2] < mjMAXVAL) {
        HCandidate* cand = measureCandidate(grid, &set, point);
        if (cand) {
          cand->eligible = shape.geom_type == mjGEOM_CAPSULE && samplePeak(grid, r, c);
          keepCandidate(&set, cand);
        }
      }
    }
  }
  // (3) ridge edges sharp enough to beat the slack: a geom whose surface curves no flatter
  // than radius R, across a kink whose triangle normals differ by 1 - cos(a), sits at most
  // about R*(1 - cos(a))/4 deeper on the edge than on either triangle. R is the bounding
  // radius, or an ellipsoid's flattest curvature (longest semi-axis squared over the
  // shortest), which exceeds it. Edges from each sample: along the row
  // (between the lower triangle of cell (r, c) and the upper one of cell (r-1, c)),
  // along the column (upper of (r, c), lower of (r, c-1)), and the cell's diagonal.
  for (int r = rmin; r <= rmax + 1; r++) {
    for (int c = cmin; c <= cmax + 1; c++) {
      for (int e = 0; e < 3; e++) {
        int r1 = r + (e > 0), c1 = c + (e != 1);
        if (r1 > rmax + 1 || c1 > cmax + 1) continue;
        if (hmax(sampleZ(grid, r, c), sampleZ(grid, r1, c1)) < lo[2] - prefilter) continue;
        // triangle a, triangle b, and b's corner off the edge
        int ra = r, ca = c, ua = 0, rb = r, cb = c, ub = 1, ro, co;
        if (e == 0) {
          rb = r - 1; ro = r - 1; co = c;
        } else if (e == 1) {
          ca = c - 1; ro = r + 1; co = c + 1;
        } else {
          ro = r + 1; co = c;
        }
        if (ra > grid->nr - 2 || rb < 0 || ca < 0 || cb > grid->nc - 2) continue;
        mjtNum na[3], nb[3];
        triangleNormal(grid, ra, ca, ua, na);
        triangleNormal(grid, rb, cb, ub, nb);
        if (flattest*(1 - hdot3(na, nb))/4 < slack) continue;
        // a ridge: b's far corner lies below a's plane
        mjtNum xe = c*grid->dx - grid->sx, ye = r*grid->dy - grid->sy, ze = sampleZ(grid, r, c);
        mjtNum xo = co*grid->dx - grid->sx, yo = ro*grid->dy - grid->sy;
        mjtNum zplane = ze - (na[0]*(xo - xe) + na[1]*(yo - ye))/na[2];
        if (sampleZ(grid, ro, co) >= zplane) continue;
        addEdgeMinimum(grid, &set, &shape, r, c, r1, c1, lo, hi);
      }
    }
  }

  int chosen[4], ncon = 0, shallow = -1;
  if (shape.geom_type == mjGEOM_CAPSULE) {
    // (4) the side across ridges, then the support along the axis
    addAxisCrossings(grid, &set, &shape);
    ncon = selectCapsule(&set, axis, slack, chosen);
    ncon = emitContacts(&set, chosen, ncon, con, d->geom_xpos + 3*hgeom, d->geom_xmat + 9*hgeom);
    // as mjc_PlaneCapsule, align the contact frames with the axis
    const mjtNum* gmat = d->geom_xmat + 9*g;
    for (int i = 0; i < ncon; i++) {
      con[i].tangent[0] = gmat[2];
      con[i].tangent[1] = gmat[5];
      con[i].tangent[2] = gmat[8];
    }
    mj_freeStack(d);
    return ncon;
  }

  // keep the plane functions' points; add the deepest feature when it is deeper than all
  // of them by the slack, replacing the shallowest point when all four slots are taken
  mjtNum deepest = mjMAXVAL;
  for (int i = 0; i < nshape; i++) {
    chosen[ncon] = i;
    if (shallow < 0 || set.list[i].dist > set.list[chosen[shallow]].dist) {
      shallow = ncon;
    }
    deepest = hmin(deepest, set.list[i].dist);
    ncon++;
  }
  int feature = -1;
  if (gatherCandidates(&set) > nshape && set.list[nshape].dist < deepest - slack) {
    feature = nshape;
  }
  if (feature >= 0) {
    if (ncon < 4) {
      chosen[ncon++] = feature;
    } else {
      chosen[shallow] = feature;
    }
  }
  ncon = emitContacts(&set, chosen, ncon, con, d->geom_xpos + 3*hgeom, d->geom_xmat + 9*hgeom);
  mj_freeStack(d);
  return ncon;
}


// Contacts over the grid interior, or -1 where the prisms apply.
static int smoothHField(const mjModel* m, mjData* d, mjPreContact* con, int hgeom, int g,
                        mjtNum margin) {
  int hid = m->geom_dataid[hgeom];
  const mjtNum* size = m->hfield_size + 4*hid;
  HGrid grid = {m->hfield_data + m->hfield_adr[hid], m->hfield_nrow[hid], m->hfield_ncol[hid],
                size[0], size[1], size[2], size[3], 0, 0, 0, 0, 0};
  if (grid.nr < 2 || grid.nc < 2) {
    return -1;
  }
  grid.dx = 2*size[0]/(grid.nc - 1);
  grid.dy = 2*size[1]/(grid.nr - 1);
  grid.idx = 1/grid.dx;
  grid.idy = 1/grid.dy;
  grid.crease = 1e-5*size[2]*hmax(grid.idx, grid.idy);  // float heights round at ~6e-8*sz
  if (grid.dx*grid.dy < 1e-12 || margin < 0 || margin > size[0] + size[1]) {
    return -1;
  }
  switch (m->geom_type[g]) {
  case mjGEOM_BOX:
  case mjGEOM_MESH:
    return smoothPolytope(m, d, con, hgeom, g, margin, &grid);
  case mjGEOM_SPHERE:
  case mjGEOM_CAPSULE:
  case mjGEOM_ELLIPSOID:
  case mjGEOM_CYLINDER:
    return smoothCurved(m, d, con, hgeom, g, margin, &grid);
  default:
    return -1;
  }
}


// Upstream's early-out: the geom's bounding sphere clears the field's box.
static int beyondField(const mjModel* m, const mjData* d, int hgeom, int g, mjtNum margin) {
  const mjtNum* size = m->hfield_size + 4*m->geom_dataid[hgeom];
  mjtNum dif[3], pos[3];
  mju_sub3(dif, d->geom_xpos + 3*g, d->geom_xpos + 3*hgeom);
  mju_mulMatTVec3(pos, d->geom_xmat + 9*hgeom, dif);
  mjtNum radius = m->geom_rbound[g] + margin;
  return pos[0] - radius > size[0] || pos[0] + radius < -size[0] ||
         pos[1] - radius > size[1] || pos[1] + radius < -size[1] ||
         pos[2] - radius > size[2] || pos[2] + radius < -size[3];
}

int mjc_SmoothHField(const mjModel* m, mjData* d, mjPreContact* con, int hgeom, int g,
                     mjtNum margin) {
  if (m->hfield_collider[m->geom_dataid[hgeom]] == 0) {
    if (beyondField(m, d, hgeom, g, margin)) {
      return 0;
    }
    int ncon = smoothHField(m, d, con, hgeom, g, margin);
    if (ncon >= 0) {
      return ncon;
    }
  }
  return mjc_ConvexHField(m, d, con, hgeom, g, margin);
}

// Copyright 2026 Feather contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

// A heightfield's cell prisms share internal side faces. Treating those faces as
// independent convex obstacles produces lateral EPA normals even on a flat field.
// For interior polyhedra, clip the lower hull surface to each top triangle's XY
// footprint instead. The resulting witnesses belong to the actual hull; their
// signed distances, midpoints, and normals all refer to the same terrain plane.
// No model/data layout changes and no writes to compiled geometry.

#include "engine/engine_collision_hfield.h"

#include <mujoco/mjmacro.h>
#include "engine/engine_memory.h"
#include "engine/engine_util_blas.h"
#include "engine/engine_util_misc.h"
#include "engine/engine_util_spatial.h"


static mjtNum distance2(const mjtNum a[3], const mjtNum b[3]) {
  mjtNum v[3];
  mju_sub3(v, a, b);
  return mju_dot3(v, v);
}


// Keep four well-separated witnesses per coplanar patch, including the deepest.
// This merges cell seams and hull-face triangulation, rather than allowing a
// flat sole's contact count and support footprint to depend on the grid density.
static int addContact(mjPreContact* con, int ncon, const mjtNum point[3],
                      const mjtNum normal[3], mjtNum plane, mjtNum margin) {
  mjPreContact candidate;
  candidate.dist = mju_dot3(point, normal) - plane;
  if (candidate.dist > margin) {
    return ncon;
  }
  mju_copy3(candidate.normal, normal);
  mju_zero3(candidate.tangent);
  mju_addScl3(candidate.pos, point, normal, -0.5*candidate.dist);

  int group[4], ngroup = 0;
  for (int i = 0; i < ncon; i++) {
    mjtNum offset = mju_dot3(con[i].pos, normal) - 0.5*con[i].dist;
    if (mju_dot3(con[i].normal, normal) > 1 - 1e-12 &&
        mju_abs(offset - plane) < 1e-9) {
      if (distance2(candidate.pos, con[i].pos) < 1e-18) {
        if (candidate.dist < con[i].dist) {
          con[i] = candidate;
        }
        return ncon;
      }
      if (ngroup < 4) {
        group[ngroup++] = i;
      }
    }
  }

  if (ngroup < 4) {
    if (ncon < mjMAXCONPAIR) {
      con[ncon++] = candidate;
    } else {
      // The engine's pair cap also applies to highly nonplanar terrain. Prefer
      // a deeper witness to dropping all later cells by traversal order.
      int shallow = 0;
      for (int i = 1; i < ncon; i++) {
        if (con[i].dist > con[shallow].dist) {
          shallow = i;
        }
      }
      if (candidate.dist < con[shallow].dist) {
        con[shallow] = candidate;
      }
    }
    return ncon;
  }

  mjPreContact points[5];
  for (int i = 0; i < 4; i++) {
    points[i] = con[group[i]];
  }
  points[4] = candidate;
  int chosen[4] = {0, -1, -1, -1};
  for (int i = 1; i < 5; i++) {
    if (points[i].dist < points[chosen[0]].dist) {
      chosen[0] = i;
    }
  }
  for (int k = 1; k < 4; k++) {
    mjtNum best = -1;
    for (int i = 0; i < 5; i++) {
      int used = 0;
      for (int j = 0; j < k; j++) {
        used |= chosen[j] == i;
      }
      if (used) {
        continue;
      }
      mjtNum score;
      if (k == 2) {
        // Third point spans area, not just a long line.
        mjtNum a[3], b[3], cross[3];
        mju_sub3(a, points[chosen[1]].pos, points[chosen[0]].pos);
        mju_sub3(b, points[i].pos, points[chosen[0]].pos);
        mju_cross(cross, a, b);
        score = mju_dot3(cross, cross);
      } else {
        score = mjMAXVAL;
        for (int j = 0; j < k; j++) {
          score = mju_min(score, distance2(points[i].pos, points[chosen[j]].pos));
        }
      }
      if (score > best) {
        best = score;
        chosen[k] = i;
      }
    }
  }
  for (int i = 0; i < 4; i++) {
    con[group[i]] = points[chosen[i]];
  }
  return ncon;
}


// Clip one hull triangle by the three vertical halfspaces of a CCW terrain
// triangle. A triangle clipped by three planes has at most six vertices.
static int clipTriangle(mjtNum output[8][3], const mjtNum* a, const mjtNum* b,
                        const mjtNum* c, mjtNum terrain[3][3]) {
  mjtNum buffers[2][8][3];
  mju_copy3(buffers[0][0], a);
  mju_copy3(buffers[0][1], b);
  mju_copy3(buffers[0][2], c);
  int count = 3;
  for (int edge = 0; edge < 3 && count; edge++) {
    mjtNum (*src)[3] = buffers[edge % 2];
    mjtNum (*dst)[3] = buffers[1 - edge % 2];
    const mjtNum* p = terrain[edge];
    const mjtNum* q = terrain[(edge + 1) % 3];
    mjtNum ex = q[0] - p[0], ey = q[1] - p[1];
    int out = 0;
    for (int i = 0; i < count; i++) {
      const mjtNum* v = src[i];
      const mjtNum* w = src[(i + 1) % count];
      mjtNum dv = ex*(v[1] - p[1]) - ey*(v[0] - p[0]);
      mjtNum dw = ex*(w[1] - p[1]) - ey*(w[0] - p[0]);
      // A vertex on the clipping plane is already retained by its incoming
      // edge. Strict crossings avoid duplicating it and preserve the bound.
      if ((dv > 0 && dw < 0) || (dv < 0 && dw > 0)) {
        mjtNum t = dv / (dv - dw);
        for (int j = 0; j < 3; j++) {
          dst[out][j] = v[j] + t*(w[j] - v[j]);
        }
        out++;
      }
      if (dw >= 0) {
        mju_copy3(dst[out++], w);
      }
    }
    count = out;
  }
  for (int i = 0; i < count; i++) {
    mju_copy3(output[i], buffers[1][i]);
  }
  return count;
}


// Intersect a lower hull triangle with the top surface, restricted to its cells.
static int collideTriangle(mjPreContact* con, int ncon, const mjtNum* a,
                           const mjtNum* b, const mjtNum* c, const mjtNum size[4],
                           const float* heights, int nr, int nc, mjtNum margin) {
  mjtNum dx = 2*size[0] / (nc - 1), dy = 2*size[1] / (nr - 1);
  mjtNum xmin = mju_min(a[0], mju_min(b[0], c[0]));
  mjtNum xmax = mju_max(a[0], mju_max(b[0], c[0]));
  mjtNum ymin = mju_min(a[1], mju_min(b[1], c[1]));
  mjtNum ymax = mju_max(a[1], mju_max(b[1], c[1]));
  mjtNum zmin = mju_min(a[2], mju_min(b[2], c[2]));
  int cmin = mjMAX(0, (int)mju_floor((xmin + size[0])/dx));
  int cmax = mjMIN(nc - 2, (int)mju_floor((xmax + size[0])/dx));
  int rmin = mjMAX(0, (int)mju_floor((ymin + size[1])/dy));
  int rmax = mjMIN(nr - 2, (int)mju_floor((ymax + size[1])/dy));
  const int corners[2][3] = {{0, 1, 2}, {0, 2, 3}};
  for (int r = rmin; r <= rmax; r++) {
    for (int col = cmin; col <= cmax; col++) {
      mjtNum x = col*dx - size[0], y = r*dy - size[1];
      mjtNum cell[4][3] = {
        {x, y, heights[r*nc + col]*size[2]},
        {x + dx, y, heights[r*nc + col + 1]*size[2]},
        {x + dx, y + dy, heights[(r + 1)*nc + col + 1]*size[2]},
        {x, y + dy, heights[(r + 1)*nc + col]*size[2]}
      };
      for (int half = 0; half < 2; half++) {
        mjtNum top[3][3], e1[3], e2[3], normal[3];
        for (int j = 0; j < 3; j++) {
          mju_copy3(top[j], cell[corners[half][j]]);
        }
        mju_sub3(e1, top[1], top[0]);
        mju_sub3(e2, top[2], top[0]);
        mju_cross(normal, e1, e2);
        mju_normalize3(normal);
        if (zmin > mju_max(top[0][2], mju_max(top[1][2], top[2][2])) +
            margin / normal[2]) {
          continue;
        }
        mjtNum plane = mju_dot3(normal, top[0]);
        mjtNum clipped[8][3];
        int count = clipTriangle(clipped, a, b, c, top);
        for (int j = 0; j < count; j++) {
          ncon = addContact(con, ncon, clipped[j], normal, plane, margin);
        }
      }
    }
  }
  return ncon;
}


int mjc_PolyHField(const mjModel* m, mjData* d, mjPreContact* con,
                  int hgeom, int pgeom, mjtNum margin) {
  int type = m->geom_type[pgeom];
  int mesh = m->geom_dataid[pgeom];
  if (type != mjGEOM_BOX && (type != mjGEOM_MESH || m->mesh_polynum[mesh] <= 0)) {
    return -1;
  }

  int hid = m->geom_dataid[hgeom];
  const mjtNum* size = m->hfield_size + 4*hid;
  const mjtNum* hpos = d->geom_xpos + 3*hgeom;
  const mjtNum* hmat = d->geom_xmat + 9*hgeom;
  const mjtNum* pmat = d->geom_xmat + 9*pgeom;
  mjtNum dif[3], pos[3], mat[9];
  mju_sub3(dif, d->geom_xpos + 3*pgeom, hpos);
  mju_mulMatTVec3(pos, hmat, dif);
  mju_mulMatTMat3(mat, hmat, pmat);
  mjtNum radius = m->geom_rbound[pgeom] + margin;
  if (pos[0] - radius > size[0] || pos[0] + radius < -size[0] ||
      pos[1] - radius > size[1] || pos[1] + radius < -size[1] ||
      pos[2] - radius > size[2] || pos[2] + radius < -size[3]) {
    return 0;
  }

  mj_markStack(d);
  int nv = type == mjGEOM_BOX ? 8 : m->mesh_vertnum[mesh];
  mjtNum* vertices = mj_stackAllocNum(d, 3*nv);
  const float* source = type == mjGEOM_MESH ? m->mesh_vert + 3*m->mesh_vertadr[mesh] : NULL;
  for (int i = 0; i < nv; i++) {
    mjtNum local[3];
    for (int j = 0; j < 3; j++) {
      local[j] = source ? source[3*i + j] : ((i & (1 << j)) ? 1 : -1)*m->geom_size[3*pgeom + j];
    }
    mju_mulMatVec3(vertices + 3*i, mat, local);
    mju_addTo3(vertices + 3*i, pos);
    // At the outer walls or underside the old solid-prism contacts are valid.
    if (vertices[3*i] <= -size[0] + margin || vertices[3*i] >= size[0] - margin ||
        vertices[3*i + 1] <= -size[1] + margin || vertices[3*i + 1] >= size[1] - margin ||
        vertices[3*i + 2] <= -size[3] + margin) {
      mj_freeStack(d);
      return -1;
    }
  }

  const float* heights = m->hfield_data + m->hfield_adr[hid];
  int nr = m->hfield_nrow[hid], nc = m->hfield_ncol[hid], ncon = 0;
  // Hull polygon vertex indices are local to the mesh, including with maxhullvert.
  // Fan triangulation avoids a fixed polygon-size limit.
  if (type == mjGEOM_MESH) {
    int first = m->mesh_polyadr[mesh];
    for (int f = first; f < first + m->mesh_polynum[mesh]; f++) {
      const mjtNum* normal = m->mesh_polynormal + 3*f;
      if (mju_dot3(mat + 6, normal) >= -1e-12) {
        continue;
      }
      const int* ids = m->mesh_polyvert + m->mesh_polyvertadr[f];
      for (int j = 1; j + 1 < m->mesh_polyvertnum[f]; j++) {
        ncon = collideTriangle(con, ncon, vertices + 3*ids[0], vertices + 3*ids[j],
                               vertices + 3*ids[j + 1], size, heights, nr, nc, margin);
      }
    }
  } else {
    const int faces[6][4] = {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4},
                             {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    for (int f = 0; f < 6; f++) {
      if ((f % 2 ? 1 : -1)*mat[6 + f/2] >= -1e-12) {
        continue;
      }
      for (int j = 1; j < 3; j++) {
        ncon = collideTriangle(con, ncon, vertices + 3*faces[f][0], vertices + 3*faces[f][j],
                               vertices + 3*faces[f][j + 1], size, heights, nr, nc, margin);
      }
    }
  }
  mj_freeStack(d);

  for (int i = 0; i < ncon; i++) {
    mjtNum point[3], normal[3];
    mju_copy3(point, con[i].pos);
    mju_copy3(normal, con[i].normal);
    mju_mulMatVec3(con[i].pos, hmat, point);
    mju_addTo3(con[i].pos, hpos);
    mju_mulMatVec3(con[i].normal, hmat, normal);
  }
  return ncon;
}

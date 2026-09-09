// Copyright 2026 Feather contributors. Licensed under Apache-2.0.
// Narrow integration helpers. Model-owned hfield caches follow the model buffer lifecycle.
#include <math.h>
#include <stddef.h>
#include <string.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <time.h>
#endif

#include <mujoco/mjfeather.h>
#include <mujoco/mujoco.h>
#include "engine/engine_collision_hfield.h"
#include "engine/engine_memory.h"
#include "engine/engine_sleep.h"
#include "engine/engine_util_blas.h"
#include "build_id.h"

const char* mjf_buildId(void) { return MJ_FEATHER_BUILD_ID; }

#ifdef _WIN32
static mjtNum timer_tick_ms;

static mjtNum timerMilliseconds(void) {
  LARGE_INTEGER ticks;
  if (!QueryPerformanceCounter(&ticks)) {
    mju_error("Feather timer: QueryPerformanceCounter failed");
  }
  return (mjtNum)ticks.QuadPart * timer_tick_ms;
}
#else
static mjtNum timerMilliseconds(void) {
  struct timespec now;
  if (clock_gettime(CLOCK_MONOTONIC, &now)) {
    mju_error("Feather timer: clock_gettime failed");
  }
  return (mjtNum)now.tv_sec * 1000 + (mjtNum)now.tv_nsec * 1e-6;
}
#endif

void mjf_enableTimers(int enable) {
#ifdef _WIN32
  if (enable) {
    LARGE_INTEGER frequency;
    if (!QueryPerformanceFrequency(&frequency) || frequency.QuadPart <= 0) {
      mju_error("Feather timer: QueryPerformanceFrequency failed");
    }
    timer_tick_ms = 1000.0 / (mjtNum)frequency.QuadPart;
  }
#endif
  mjcb_time = enable ? timerMilliseconds : NULL;
}

int mjf_layout(const char* type, const char* field) {
  if (!type) return -1;
#define SIZE(T) if (!strcmp(type, #T) && !field) return (int)sizeof(T)
#define OFFSET(T, F) if (!strcmp(type, #T) && field && !strcmp(field, #F)) \
    return (int)offsetof(T, F)
  SIZE(mjtNum); SIZE(mjModel); SIZE(mjData); SIZE(mjOption); SIZE(mjContact);
  OFFSET(mjModel, opt); OFFSET(mjModel, nq); OFFSET(mjModel, body_mass);
  OFFSET(mjModel, geom_type); OFFSET(mjModel, geom_size); OFFSET(mjModel, hfield_data);
  OFFSET(mjModel, hfield_mipmin); OFFSET(mjModel, hfield_mipmax); OFFSET(mjModel, hfield_mipvalid);
  OFFSET(mjData, qpos); OFFSET(mjData, qvel); OFFSET(mjData, contact);
  OFFSET(mjData, xpos); OFFSET(mjData, xfrc_applied); OFFSET(mjData, tree_asleep);
  OFFSET(mjContact, dist); OFFSET(mjContact, pos); OFFSET(mjContact, frame);
  OFFSET(mjContact, geom); OFFSET(mjOption, timestep); OFFSET(mjOption, integrator);
  OFFSET(mjOption, iterations); OFFSET(mjOption, enableflags);
#undef SIZE
#undef OFFSET
  return -1;
}

int mjf_wakeAll(const mjModel* m, mjData* d) {
  if (!m || !d) return -1;
  if (!(m->opt.enableflags & mjENBL_SLEEP)) return 0;
  int woke = 0;
  for (int i = 0; i < m->ntree; i++) {
    if (d->tree_asleep[i] >= 0) {
      woke += mj_wakeIsland(d->tree_asleep, m->ntree, i, -(1 + mjMINAWAKE), "model edit", d->time);
    }
  }
  if (woke) mj_updateSleep(m, d);
  return woke;
}

int mjf_writeHfieldData(mjModel* m, mjData* d, int id, const float* values, int count) {
  if (!m || id < 0 || id >= m->nhfield) return -1;
  return mjf_writeHfieldRows(m, d, id, 0, m->hfield_nrow[id], values, count);
}

int mjf_enableHfieldMip(mjModel* m, int id, int enable) {
  if (!m || id < 0 || id >= m->nhfield) return -1;
  if (!enable) {
    m->hfield_mipvalid[id] = 0;
    return 0;
  }
  const float* heights = m->hfield_data + m->hfield_adr[id];
  mjtSize n = (mjtSize)m->hfield_nrow[id]*m->hfield_ncol[id];
  for (mjtSize i = 0; i < n; i++) {
    if (!isfinite(heights[i])) {
      m->hfield_mipvalid[id] = 0;
      return -1;
    }
  }
  mjc_updateHFieldMip(m, id, 0, m->hfield_nrow[id]-1, 0, m->hfield_ncol[id]-1);
  m->hfield_mipvalid[id] = 1;
  return 0;
}

// The optional dirty mask identifies actual changed samples within written rows.
// This avoids waking the rack between two disjoint prints in the same update.
// Without a mask the public API conservatively treats the rectangle as changed.
static int wakeHfieldRegion(const mjModel* m, mjData* d, int id,
                            int rmin, int rmax, int cmin, int cmax,
                            const mjtByte* dirty, int dirty_row_start) {
  if (!(m->opt.enableflags & mjENBL_SLEEP)) return 0;
  int nr = m->hfield_nrow[id], nc = m->hfield_ncol[id];
  const mjtNum* size = m->hfield_size + 4*id;
  mjtNum dx = 2*size[0]/(nc-1), dy = 2*size[1]/(nr-1);
  // Explicit pairs may author a larger margin than either geom. Taking their
  // maximum is conservative and keeps this rare model-edit path simple.
  mjtNum pair_margin = 0;
  for (int p = 0; p < m->npair; p++) pair_margin = mju_max(pair_margin, m->pair_margin[p]);
  int woke = 0;
  for (int hg = 0; hg < m->ngeom; hg++) {
    if (m->geom_type[hg] != mjGEOM_HFIELD || m->geom_dataid[hg] != id) continue;
    const mjtNum* hpos = d->geom_xpos+3*hg;
    const mjtNum* hmat = d->geom_xmat+9*hg;
    for (int g = 0; g < m->ngeom; g++) {
      int tree = m->body_treeid[m->geom_bodyid[g]];
      if (tree < 0 || d->tree_asleep[tree] < 0) continue;
      int r0 = rmin, r1 = rmax, c0 = cmin, c1 = cmax;
      if (m->geom_type[g] != mjGEOM_PLANE) {
        const mjtNum* aabb = m->geom_aabb+6*g;
        const mjtNum* mat = d->geom_xmat+9*g;
        mjtNum center[3], dif[3], local[3], rotation[9];
        mju_mulMatVec3(center, mat, aabb);
        mju_addTo3(center, d->geom_xpos+3*g);
        mju_sub3(dif, center, hpos);
        mju_mulMatTVec3(local, hmat, dif);
        mju_mulMatTMat3(rotation, hmat, mat);
        mjtNum margin = mju_max(pair_margin, mju_max(m->geom_margin[g], m->geom_margin[hg]));
        if (m->opt.enableflags & mjENBL_OVERRIDE) margin = m->opt.o_margin;
        margin = mju_max(0, margin);
        mjtNum ex = margin, ey = margin;
        for (int j = 0; j < 3; j++) {
          ex += mju_abs(rotation[j])*aabb[3+j];
          ey += mju_abs(rotation[3+j])*aabb[3+j];
        }
        // Each edited vertex affects the cells on both sides. Work in the
        // hfield's own frame, including translated/rotated/instanced fields.
        mjtNum x0 = (local[0]-ex+size[0])/dx, x1 = (local[0]+ex+size[0])/dx;
        mjtNum y0 = (local[1]-ey+size[1])/dy, y1 = (local[1]+ey+size[1])/dy;
        if (x1+1 < cmin || x0-1 > cmax || y1+1 < rmin || y0-1 > rmax) continue;
        c0 = mjMAX(cmin, (int)ceil(mju_max(-1, x0-1e-9))-1);
        c1 = mjMIN(cmax, (int)mju_floor(mju_min(nc, x1+1e-9))+1);
        r0 = mjMAX(rmin, (int)ceil(mju_max(-1, y0-1e-9))-1);
        r1 = mjMIN(rmax, (int)mju_floor(mju_min(nr, y1+1e-9))+1);
      }
      int hit = !dirty && r0 <= r1 && c0 <= c1;
      for (int r = r0; dirty && !hit && r <= r1; r++) {
        for (int c = c0; c <= c1; c++) {
          if (dirty[(mjtSize)(r-dirty_row_start)*nc+c]) { hit = 1; break; }
        }
      }
      if (hit) {
        // Wake the complete upstream sleeping island, including welded or
        // contact/equality-coupled trees, never just one member of its cycle.
        woke += mj_wakeIsland(d->tree_asleep, m->ntree, tree, -(1+mjMINAWAKE),
                              "heightfield edit", d->time);
      }
    }
  }
  if (woke) mj_updateSleep(m, d);
  return woke;
}

int mjf_wakeHfieldRegion(const mjModel* m, mjData* d, int id,
                         int row_start, int row_count, int col_start, int col_count) {
  if (!m || !d || id < 0 || id >= m->nhfield || row_start < 0 || col_start < 0 ||
      row_count < 0 || col_count < 0 || row_start > m->hfield_nrow[id] ||
      col_start > m->hfield_ncol[id] || row_count > m->hfield_nrow[id]-row_start ||
      col_count > m->hfield_ncol[id]-col_start) return -1;
  if (!row_count || !col_count) return 0;
  return wakeHfieldRegion(m, d, id, row_start, row_start+row_count-1,
                          col_start, col_start+col_count-1, NULL, 0);
}

int mjf_writeHfieldRows(mjModel* m, mjData* d, int id, int row_start,
                       int row_count, const float* values, int count) {
  if (!m || !d || !values || id < 0 || id >= m->nhfield || row_start < 0 ||
      row_count < 0 || row_start > m->hfield_nrow[id] ||
      row_count > m->hfield_nrow[id]-row_start ||
      count != (mjtSize)row_count*m->hfield_ncol[id]) return -1;
  for (int i = 0; i < count; i++) {
    if (!isfinite(values[i])) return -1;
  }
  int nc = m->hfield_ncol[id];
  mjtByte* dirty = NULL;
  if (count && (m->opt.enableflags & mjENBL_SLEEP)) {
    mj_markStack(d);
    dirty = mj_stackAllocByte(d, count, 1);
    memset(dirty, 0, count);
  }
  float* dst = m->hfield_data + m->hfield_adr[id] + (mjtSize)row_start*nc;
  int rmin = m->hfield_nrow[id], rmax = -1, cmin = nc, cmax = -1;
  for (int i = 0; i < count; i++) {
    float v = values[i] < 0 ? 0 : values[i] > 1 ? 1 : values[i];
    if (dst[i] != v) {
      if (dirty) dirty[i] = 1;
      dst[i] = v;
      int r = row_start+i/nc, c = i%nc;
      rmin = mjMIN(rmin, r); rmax = mjMAX(rmax, r);
      cmin = mjMIN(cmin, c); cmax = mjMAX(cmax, c);
    }
  }
  if (rmax >= 0) {
    if (m->hfield_mipvalid[id]) mjc_updateHFieldMip(m, id, rmin, rmax, cmin, cmax);
    wakeHfieldRegion(m, d, id, rmin, rmax, cmin, cmax, dirty, row_start);
  }
  if (dirty) mj_freeStack(d);
  return rmax >= 0;
}

int mjf_bodyVelocities(const mjModel* m, mjData* const* datas, int ndata,
                       const int* bodies, int nbody, int object_type, mjtNum* out) {
  if (!m || ndata < 0 || nbody < 0 ||
      (object_type != mjOBJ_BODY && object_type != mjOBJ_XBODY) ||
      (nbody && !bodies) || (ndata && (!datas || (nbody && !out)))) return -1;
  for (int i = 0; i < ndata; i++) if (!datas[i]) return -1;
  for (int b = 0; b < nbody; b++) if (bodies[b] < 0 || bodies[b] >= m->nbody) return -1;
  for (int i = 0; i < ndata; i++) {
    for (int b = 0; b < nbody; b++) {
      mj_objectVelocity(m, datas[i], object_type, bodies[b], out, 0);
      out += 6;
    }
  }
  return 0;
}

static int consumeDivergence(mjData* d) {
  int result = 0;
  for (int w = mjWARN_BADQPOS; w <= mjWARN_BADQACC; w++) {
    result |= d->warning[w].number != 0;
    d->warning[w].number = 0;
  }
  return result;
}

static mjtNum clamp(mjtNum v, mjtNum cap) {
  return v < -cap ? -cap : v > cap ? cap : v;
}

int mjf_stepGuarded(const mjModel* m, mjData* d, int nstep,
                    const int* joint_dofs, const mjtNum* limits, int njoint,
                    int root_dof, mjtNum root_cap, int impulse_body,
                    const mjtNum* impulse_force) {
  if (!m || !d || nstep < 0 || njoint < 0 ||
      (njoint && (!joint_dofs || !limits)) || root_dof < 0 || root_dof + 6 > m->nv ||
      !isfinite(root_cap) || root_cap <= 0 || impulse_body < -1 || impulse_body >= m->nbody ||
      (impulse_body >= 0 && !impulse_force)) return -1;
  for (int j = 0; j < njoint; j++) {
    if (joint_dofs[j] < 0 || joint_dofs[j] >= m->nv ||
        !isfinite(limits[j]) || limits[j] < 0) return -1;
  }
  if (impulse_body >= 0) {
    for (int j = 0; j < 3; j++) if (!isfinite(impulse_force[j])) return -1;
  }
  if (!nstep) return 0;
  int diverged = consumeDivergence(d);
  for (int s = 0; s < nstep; s++) {
    int impulse = s == 0 && impulse_body >= 0;
    if (impulse) {
      for (int j = 0; j < 3; j++) d->xfrc_applied[6*impulse_body + j] += impulse_force[j];
    }
    mj_step(m, d);
    int reset = consumeDivergence(d);
    diverged |= reset;
    if (impulse && (!reset || (m->opt.disableflags & mjDSBL_AUTORESET))) {
      for (int j = 0; j < 3; j++) d->xfrc_applied[6*impulse_body + j] -= impulse_force[j];
    }
    for (int j = 0; j < njoint; j++) {
      int v = joint_dofs[j];
      d->qvel[v] = clamp(d->qvel[v], limits[j]);
    }
    for (int j = 0; j < 6; j++) d->qvel[root_dof + j] = clamp(d->qvel[root_dof + j], root_cap);
  }
  return diverged;
}

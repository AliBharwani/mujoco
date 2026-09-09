// Copyright 2026 Feather contributors. Licensed under Apache-2.0.
// Narrow integration helpers: no public model/data layout changes.
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
#include "engine/engine_sleep.h"
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
  if (!m || !d || !values || id < 0 || id >= m->nhfield ||
      count != m->hfield_nrow[id]*m->hfield_ncol[id]) return -1;
  for (int i = 0; i < count; i++) {
    if (!isfinite(values[i])) return -1;
  }
  float* dst = m->hfield_data + m->hfield_adr[id];
  int changed = 0;
  for (int i = 0; i < count; i++) {
    float v = values[i] < 0 ? 0 : values[i] > 1 ? 1 : values[i];
    if (dst[i] != v) { dst[i] = v; changed = 1; }
  }
  if (changed) mjf_wakeAll(m, d);
  return changed;
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

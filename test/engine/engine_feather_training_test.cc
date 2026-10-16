// Copyright 2026 Feather contributors. Licensed under Apache-2.0.

// Tests for feather's training helpers (integration16, mjfeather.h): mjf_stepGuarded,
// mjf_bodyVelocities, and the query names before integration14, mjf_raycastFiltered and
// mjf_overlapSphereFiltered. Each is held to the plain calls it stands for.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjfeather.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using FeatherTrainingTest = MujocoTest;

// a free box over a plane with a two-hinge arm: the shape of feather's pooled worlds
static const char* const kModel = R"(
<mujoco>
  <option timestep="0.002" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 0.1"/>
    <body name="root" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1" pos="0.03 0 0" mass="1"/>
      <body name="upper" pos="0 0 0.3">
        <joint name="j1" axis="0 1 0" damping="0.1"/>
        <geom size="0.1" mass="0.5"/>
        <body name="lower" pos="0 0 0.3">
          <joint name="j2" axis="1 0 0" damping="0.1"/>
          <geom size="0.1" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="j1" gear="5"/>
    <motor joint="j2" gear="5"/>
  </actuator>
</mujoco>
)";

static bool SameBytes(const mjtNum* a, const mjtNum* b, int n) {
  return std::memcmp(a, b, sizeof(mjtNum) * n) == 0;
}

static bool SameState(const mjModel* m, const mjData* a, const mjData* b) {
  return SameBytes(a->qpos, b->qpos, m->nq) && SameBytes(a->qvel, b->qvel, m->nv) &&
         SameBytes(a->qacc, b->qacc, m->nv) && SameBytes(a->xfrc_applied, b->xfrc_applied,
                                                         6 * m->nbody);
}

static mjtNum Clamp(mjtNum v, mjtNum cap) { return v < -cap ? -cap : v > cap ? cap : v; }

class Worlds {
 public:
  explicit Worlds(const mjModel* m) : m_(m) {
    for (int i = 0; i < 2; i++) {
      d_[i] = mj_makeData(m);
      d_[i]->ctrl[0] = 0.7;
      d_[i]->ctrl[1] = -0.4;
      d_[i]->qvel[0] = 0.3;
      d_[i]->qvel[5] = 2.0;
      d_[i]->qvel[6] = -3.0;
    }
  }
  ~Worlds() {
    mj_deleteData(d_[0]);
    mj_deleteData(d_[1]);
  }
  mjData* guarded() { return d_[0]; }
  mjData* plain() { return d_[1]; }

 private:
  const mjModel* m_;
  mjData* d_[2];
};

TEST_F(FeatherTrainingTest, StepGuardedIsMjStepWhenNothingBinds) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  const mjModel* m = model.get();
  Worlds w(m);
  const int dofs[2] = {6, 7};
  const mjtNum limits[2] = {1e6, 1e6};
  EXPECT_EQ(mjf_stepGuarded(m, w.guarded(), 300, dofs, limits, 2, 0, 1e6, -1, nullptr), 0);
  for (int s = 0; s < 300; s++) {
    mj_step(m, w.plain());
  }
  EXPECT_TRUE(SameState(m, w.guarded(), w.plain()));
  EXPECT_EQ(mjf_stepGuarded(m, w.guarded(), 0, dofs, limits, 2, 0, 1e6, -1, nullptr), 0);
  EXPECT_TRUE(SameState(m, w.guarded(), w.plain()));
}

TEST_F(FeatherTrainingTest, StepGuardedClampsAfterEachSubstep) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  const mjModel* m = model.get();
  Worlds w(m);
  const int dofs[2] = {7, 6};
  const mjtNum limits[2] = {0.5, 0.25};
  const mjtNum cap = 0.2;
  EXPECT_EQ(mjf_stepGuarded(m, w.guarded(), 20, dofs, limits, 2, 0, cap, -1, nullptr), 0);
  mjData* d = w.plain();
  for (int s = 0; s < 20; s++) {
    mj_step(m, d);
    for (int j = 0; j < 2; j++) d->qvel[dofs[j]] = Clamp(d->qvel[dofs[j]], limits[j]);
    for (int j = 0; j < 6; j++) d->qvel[j] = Clamp(d->qvel[j], cap);
  }
  EXPECT_TRUE(SameState(m, w.guarded(), d));
  for (int j = 0; j < 6; j++) EXPECT_LE(std::abs(w.guarded()->qvel[j]), cap);
}

TEST_F(FeatherTrainingTest, StepGuardedImpulseLandsOnTheFirstSubstepOnly) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  const mjModel* m = model.get();
  Worlds w(m);
  const mjtNum force[3] = {30, -10, 5};
  const int body = 1;
  w.guarded()->xfrc_applied[6*2 + 3] = w.plain()->xfrc_applied[6*2 + 3] = 0.25;  // persistent
  EXPECT_EQ(mjf_stepGuarded(m, w.guarded(), 4, nullptr, nullptr, 0, 0, 1e6, body, force), 0);
  mjData* d = w.plain();
  for (int s = 0; s < 4; s++) {
    if (!s) for (int j = 0; j < 3; j++) d->xfrc_applied[6*body + j] += force[j];
    mj_step(m, d);
    if (!s) for (int j = 0; j < 3; j++) d->xfrc_applied[6*body + j] -= force[j];
  }
  EXPECT_TRUE(SameState(m, w.guarded(), d));
  for (int j = 0; j < 3; j++) EXPECT_EQ(w.guarded()->xfrc_applied[6*body + j], 0);
  EXPECT_EQ(w.guarded()->xfrc_applied[6*2 + 3], 0.25);
}

TEST_F(FeatherTrainingTest, StepGuardedConsumesDivergence) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  const mjModel* m = model.get();
  Worlds w(m);
  const mjtNum force[3] = {30, -10, 5};
  MockWarningHandler warning_handler;
  warning_handler.ExpectWarnings("Nan, Inf or huge value in QVEL");
  w.guarded()->qvel[6] = std::numeric_limits<mjtNum>::quiet_NaN();
  EXPECT_EQ(mjf_stepGuarded(m, w.guarded(), 2, nullptr, nullptr, 0, 0, 1e6, 1, force), 1);
  for (int k = mjWARN_BADQPOS; k <= mjWARN_BADQACC; k++) {
    EXPECT_EQ(w.guarded()->warning[k].number, 0);
  }
  for (int j = 0; j < 6 * m->nbody; j++) {
    EXPECT_EQ(w.guarded()->xfrc_applied[j], 0);  // the reset cleared the impulse; never undone
  }
  for (int j = 0; j < m->nv; j++) EXPECT_TRUE(std::isfinite(w.guarded()->qvel[j]));
}

TEST_F(FeatherTrainingTest, StepGuardedRefusesInvalidArgumentsWithoutStepping) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  const mjModel* m = model.get();
  Worlds w(m);
  mjData* d = w.guarded();
  const int dofs[2] = {6, 7};
  const int baddofs[2] = {6, 99};
  const mjtNum limits[2] = {1, 1};
  const mjtNum badlimits[2] = {1, -1};
  const mjtNum nan = std::numeric_limits<mjtNum>::quiet_NaN();
  const mjtNum badforce[3] = {nan, 0, 0};
  const mjtNum force[3] = {1, 0, 0};
  EXPECT_EQ(mjf_stepGuarded(nullptr, d, 1, dofs, limits, 2, 0, 1, -1, nullptr), -1);
  EXPECT_EQ(mjf_stepGuarded(m, d, -1, dofs, limits, 2, 0, 1, -1, nullptr), -1);
  EXPECT_EQ(mjf_stepGuarded(m, d, 1, nullptr, limits, 2, 0, 1, -1, nullptr), -1);
  EXPECT_EQ(mjf_stepGuarded(m, d, 1, baddofs, limits, 2, 0, 1, -1, nullptr), -1);
  EXPECT_EQ(mjf_stepGuarded(m, d, 1, dofs, badlimits, 2, 0, 1, -1, nullptr), -1);
  EXPECT_EQ(mjf_stepGuarded(m, d, 1, dofs, limits, 2, 3, 1, -1, nullptr), -1);
  EXPECT_EQ(mjf_stepGuarded(m, d, 1, dofs, limits, 2, 0, nan, -1, nullptr), -1);
  EXPECT_EQ(mjf_stepGuarded(m, d, 1, dofs, limits, 2, 0, 0, -1, nullptr), -1);
  EXPECT_EQ(mjf_stepGuarded(m, d, 1, dofs, limits, 2, 0, 1, m->nbody, force), -1);
  EXPECT_EQ(mjf_stepGuarded(m, d, 1, dofs, limits, 2, 0, 1, 1, nullptr), -1);
  EXPECT_EQ(mjf_stepGuarded(m, d, 1, dofs, limits, 2, 0, 1, 1, badforce), -1);
  EXPECT_TRUE(SameState(m, d, w.plain()));
  EXPECT_EQ(d->time, 0);
}

TEST_F(FeatherTrainingTest, BodyVelocitiesAreObjectVelocities) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  const mjModel* m = model.get();
  Worlds w(m);
  for (int s = 0; s < 40; s++) mj_step(m, w.guarded());
  for (int s = 0; s < 70; s++) mj_step(m, w.plain());
  mjData* datas[2] = {w.guarded(), w.plain()};
  const int bodies[3] = {3, 1, 2};
  for (int type : {mjOBJ_BODY, mjOBJ_XBODY}) {
    std::vector<mjtNum> out(2 * 3 * 6, -1);
    EXPECT_EQ(mjf_bodyVelocities(m, datas, 2, bodies, 3, type, out.data()), 0);
    for (int i = 0; i < 2; i++) {
      for (int b = 0; b < 3; b++) {
        mjtNum expected[6];
        mj_objectVelocity(m, datas[i], type, bodies[b], expected, 0);
        EXPECT_TRUE(SameBytes(out.data() + 6 * (3*i + b), expected, 6)) << type << i << b;
      }
    }
  }
  std::vector<mjtNum> out(6, 7);
  const int badbody[1] = {m->nbody};
  EXPECT_EQ(mjf_bodyVelocities(m, datas, 1, bodies, 1, mjOBJ_GEOM, out.data()), -1);
  EXPECT_EQ(mjf_bodyVelocities(m, datas, 1, badbody, 1, mjOBJ_BODY, out.data()), -1);
  EXPECT_EQ(mjf_bodyVelocities(m, nullptr, 1, bodies, 1, mjOBJ_BODY, out.data()), -1);
  EXPECT_EQ(mjf_bodyVelocities(m, datas, 1, bodies, 1, mjOBJ_BODY, nullptr), -1);
  for (mjtNum v : out) EXPECT_EQ(v, 7);
  EXPECT_EQ(mjf_bodyVelocities(m, datas, 0, bodies, 1, mjOBJ_BODY, nullptr), 0);
}

TEST_F(FeatherTrainingTest, FilteredQueriesAreTheMaskedOnes) {
  std::string geoms;
  for (int i = 0; i < 12; i++) {
    geoms += "<geom type=\"sphere\" size=\"0.05\" pos=\"" + std::to_string(0.03*i) + " 0 0.1\"" +
             (i % 3 ? "" : " contype=\"2\" conaffinity=\"2\"") + "/>";
  }
  std::string xml = "<mujoco><worldbody><geom name=\"floor\" type=\"plane\" size=\"5 5 .1\"/>"
                    "<geom name=\"parked\" type=\"plane\" pos=\"0 0 .05\" size=\"5 5 .1\" "
                    "contype=\"0\" conaffinity=\"0\"/>" + geoms +
                    "<body pos=\"0 0 1\"><freejoint/><geom type=\"box\" size=\".1 .2 .3\"/>"
                    "</body></worldbody></mujoco>";
  char error[1024];
  MjModelPtr model = LoadModelFromString(xml, error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  const mjModel* m = model.get();
  MjDataPtr data = MakeData(model);
  mjData* d = data.get();
  mj_forward(m, d);

  const mjtNum origins[4][3] = {{2, 2, 1}, {0.1, 0, 2}, {0.12, 0.3, 0.1}, {0, 0, 1}};
  const mjtNum dirs[4][3] = {{0, 0, -2}, {0, 0, -1}, {0, -1, 0}, {0, 0, 0}};
  for (int mask : {-1, 1, 2, 0}) {
    for (int bodyexclude : {-1, 1}) {
      for (int k = 0; k < 4; k++) {
        mjfRayHit hit;
        mjfRayHitNoNormal old;
        std::memset(&old, 0x5a, sizeof(old));
        int a = mjf_raycast(m, d, origins[k], dirs[k], bodyexclude, 1, mask, &hit);
        int b = mjf_raycastFiltered(m, d, origins[k], dirs[k], bodyexclude, 1, mask, &old);
        EXPECT_EQ(a, b) << mask << " " << k;
        EXPECT_EQ(hit.dist, old.dist);
        EXPECT_TRUE(SameBytes(hit.point, old.point, 3));
        EXPECT_EQ(hit.geom, old.geom);
        EXPECT_EQ(hit.body, old.body);
        EXPECT_EQ(mjf_raycastFiltered(m, d, origins[k], dirs[k], bodyexclude, 1, mask, nullptr),
                  a);
      }
      int small[4], big[32], smallold[4], bigold[32];
      int n = mjf_overlapSphere(m, d, origins[2], 0.3, bodyexclude, mask, small, 4);
      EXPECT_EQ(mjf_overlapSphereFiltered(m, d, origins[2], 0.3, bodyexclude, mask, smallold, 4),
                n);
      EXPECT_EQ(mjf_overlapSphere(m, d, origins[2], 0.3, bodyexclude, mask, big, 32), n);
      EXPECT_EQ(mjf_overlapSphereFiltered(m, d, origins[2], 0.3, bodyexclude, mask, bigold, 32),
                n);
      for (int i = 0; i < std::min(n, 4); i++) EXPECT_EQ(small[i], smallold[i]);
      for (int i = 0; i < std::min(n, 32); i++) EXPECT_EQ(big[i], bigold[i]);
    }
  }
  mjfRayHitNoNormal old;
  EXPECT_EQ(mjf_raycastFiltered(m, d, origins[0], dirs[0], -1, 1, -1, &old), 1);
  EXPECT_EQ(old.geom, 0);
  EXPECT_DOUBLE_EQ(old.dist, 1.0);
  EXPECT_EQ(mjf_raycastFiltered(m, d, origins[3], dirs[3], -1, 1, -1, &old), 0);
  EXPECT_EQ(old.geom, -1);
  EXPECT_EQ(old.body, -1);
  EXPECT_EQ(old.dist, -1);
}

}  // namespace
}  // namespace mujoco

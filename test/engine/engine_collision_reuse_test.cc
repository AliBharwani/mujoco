// Copyright 2026 Feather contributors. Licensed under Apache-2.0.

// Tests for engine/engine_collision_reuse.c (mjf_setContactReuse).

#include <cmath>
#include <cstring>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mjfeather.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using ContactReuseTest = MujocoTest;

// a box resting on a flat heightfield, and a capsule falling beside it
static const char* const kModel = R"(
<mujoco>
  <asset>
    <hfield name="sand" nrow="21" ncol="21" size="1 1 0.2 0.1"/>
  </asset>
  <worldbody>
    <geom name="sand" type="hfield" hfield="sand"/>
    <body name="box" pos="0 0 0.06">
      <freejoint/>
      <geom name="box" type="box" size="0.05 0.05 0.05"/>
    </body>
  </worldbody>
</mujoco>
)";

static long long Replays(const mjData* d) {
  long long stats[4];
  EXPECT_EQ(mjf_contactReuseStats(d, stats), 0);
  return stats[1];
}

// the box settled on the sand
static MjDataPtr Settled(const MjModelPtr& m) {
  MjDataPtr d = MakeData(m);
  for (int i = 0; i < 500; i++) {
    mj_step(m.get(), d.get());
  }
  mj_forward(m.get(), d.get());
  return d;
}

TEST_F(ContactReuseTest, OffUntilEnabled) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = MakeData(m);
  EXPECT_EQ(d->contactcache, nullptr);
  long long stats[4];
  EXPECT_EQ(mjf_contactReuseStats(d.get(), stats), -1);

  EXPECT_EQ(mjf_setContactReuse(d.get(), 1, -1, 0), -1);
  EXPECT_EQ(mjf_setContactReuse(d.get(), 1, NAN, 0), -1);
  EXPECT_EQ(mjf_setContactReuse(d.get(), 1, 1e-4, 2), -1);
  EXPECT_EQ(d->contactcache, nullptr);

  EXPECT_EQ(mjf_setContactReuse(d.get(), 1, 1e-4, 0), 0);
  EXPECT_NE(d->contactcache, nullptr);
  EXPECT_EQ(mjf_setContactReuse(d.get(), 0, 0, 0), 0);
  EXPECT_EQ(d->contactcache, nullptr);
}

TEST_F(ContactReuseTest, ReplayAtTheSamePoseMatchesTheCollision) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = Settled(m);
  int ncon = d->ncon;
  ASSERT_GT(ncon, 0);
  std::vector<mjContact> fresh(d->contact, d->contact + ncon);

  ASSERT_EQ(mjf_setContactReuse(d.get(), 1, 1e-4, 0), 0);
  mj_forward(m.get(), d.get());  // collides and records
  EXPECT_EQ(Replays(d.get()), 0);
  mj_forward(m.get(), d.get());  // replays
  EXPECT_EQ(Replays(d.get()), 1);

  ASSERT_EQ(d->ncon, ncon);
  for (int i = 0; i < ncon; i++) {
    EXPECT_NEAR(d->contact[i].dist, fresh[i].dist, 1e-12);
    for (int k = 0; k < 3; k++) {
      EXPECT_NEAR(d->contact[i].pos[k], fresh[i].pos[k], 1e-12);
      EXPECT_NEAR(d->contact[i].frame[k], fresh[i].frame[k], 1e-12);
    }
    EXPECT_EQ(d->contact[i].geom[0], fresh[i].geom[0]);
    EXPECT_EQ(d->contact[i].geom[1], fresh[i].geom[1]);
    EXPECT_EQ(d->contact[i].dim, fresh[i].dim);
  }
}

TEST_F(ContactReuseTest, DepthFollowsTheAnchorsAndMotionPastToleranceCollides) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = Settled(m);
  ASSERT_EQ(mjf_setContactReuse(d.get(), 1, 1e-4, 0), 0);
  mj_forward(m.get(), d.get());
  std::vector<mjtNum> dist0;
  for (int i = 0; i < d->ncon; i++) dist0.push_back(d->contact[i].dist);

  // sink 50 microns: replayed, every depth 50 microns deeper
  d->qpos[2] -= 5e-5;
  mj_forward(m.get(), d.get());
  EXPECT_EQ(Replays(d.get()), 1);
  ASSERT_EQ(d->ncon, (int)dist0.size());
  for (int i = 0; i < d->ncon; i++) {
    EXPECT_NEAR(d->contact[i].dist, dist0[i] - 5e-5, 1e-9);
  }

  // another 100 microns from the recorded pose: past the tolerance, collides
  d->qpos[2] -= 1e-4;
  mj_forward(m.get(), d.get());
  EXPECT_EQ(Replays(d.get()), 1);

  // a rotation whose chord at the box's bounding radius (8.7 cm) exceeds the tolerance also
  // collides: 2 mrad moves the box's corners 173 microns, 1 mrad (87 microns) would replay
  mjtNum axis[3] = {1, 0, 0};
  mjtNum dq[4], q[4];
  mju_axisAngle2Quat(dq, axis, 2e-3);
  mju_mulQuat(q, d->qpos + 3, dq);
  mju_copy4(d->qpos + 3, q);
  mj_forward(m.get(), d.get());
  EXPECT_EQ(Replays(d.get()), 1);
}

TEST_F(ContactReuseTest, SandWrittenUnderThePairForcesACollision) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr d = Settled(m);
  ASSERT_EQ(mjf_setContactReuse(d.get(), 1, 1e-4, 0), 0);
  mj_forward(m.get(), d.get());

  // rows far from the box: the pair still replays
  int nc = m->hfield_ncol[0];
  std::vector<float> rows(2 * nc, 0.01f);
  ASSERT_EQ(mjf_writeHfieldRows(m.get(), d.get(), 0, 0, 2, rows.data(), 2 * nc), 1);
  mj_forward(m.get(), d.get());
  EXPECT_EQ(Replays(d.get()), 1);

  // raise the rows under the box by 2 mm: it collides again and sees the new surface
  mjtNum dist_before = d->contact[0].dist;
  std::vector<float> under(3 * nc, 0.01f);
  ASSERT_EQ(mjf_writeHfieldRows(m.get(), d.get(), 0, 9, 3, under.data(), 3 * nc), 1);
  mj_forward(m.get(), d.get());
  EXPECT_EQ(Replays(d.get()), 1);
  ASSERT_GT(d->ncon, 0);
  EXPECT_LT(d->contact[0].dist, dist_before - 1e-3);
}

TEST_F(ContactReuseTest, CopyCarriesTheCacheAndResetEmptiesIt) {
  char error[1024];
  MjModelPtr m = LoadModelFromString(kModel, error, sizeof(error));
  ASSERT_THAT(m.get(), NotNull()) << error;
  MjDataPtr a = Settled(m);
  ASSERT_EQ(mjf_setContactReuse(a.get(), 1, 1e-4, mjfREUSE_EMPTY), 0);
  for (int i = 0; i < 20; i++) mj_step(m.get(), a.get());

  // a copy steps exactly as the source
  MjDataPtr b = MakeData(m);
  mj_copyData(b.get(), m.get(), a.get());
  ASSERT_NE(b->contactcache, nullptr);
  EXPECT_NE(b->contactcache, a->contactcache);
  for (int i = 0; i < 50; i++) {
    mj_step(m.get(), a.get());
    mj_step(m.get(), b.get());
  }
  EXPECT_EQ(std::memcmp(a->qpos, b->qpos, sizeof(mjtNum) * m->nq), 0);
  EXPECT_EQ(std::memcmp(a->qvel, b->qvel, sizeof(mjtNum) * m->nv), 0);
  EXPECT_GT(Replays(b.get()), 0);

  // copying a data without reuse turns it off in the destination
  MjDataPtr c = MakeData(m);
  mj_copyData(b.get(), m.get(), c.get());
  EXPECT_EQ(b->contactcache, nullptr);

  // reset keeps reuse on but forgets every pair: the first pass collides
  mj_resetData(m.get(), a.get());
  ASSERT_NE(a->contactcache, nullptr);
  long long before = Replays(a.get());
  mj_forward(m.get(), a.get());
  EXPECT_EQ(Replays(a.get()), before);
}

}  // namespace
}  // namespace mujoco

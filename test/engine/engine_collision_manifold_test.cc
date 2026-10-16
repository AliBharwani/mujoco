// Copyright 2026 Feather contributors. Licensed under Apache-2.0.

// Tests for the fork's positive-margin convex manifolds (engine_collision_gjk.c): with a margin, a
// box or mesh face resting on another clips the physical faces instead of taking upstream's
// perturbed-pose path, at the margin's boundary as well, and falls back as upstream does where no
// faces oppose or a flag selects another path.

#include <array>
#include <cmath>
#include <set>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using ConvexManifoldTest = MujocoTest;

std::string Corners(double x, double y, double z) {
  std::string s;
  for (double a : {-x, x}) for (double b : {-y, y}) for (double c : {-z, z}) {
    s += std::to_string(a) + " " + std::to_string(b) + " " + std::to_string(c) + " ";
  }
  return s;
}

std::string Geom(bool mesh, const char* name, const std::string& size) {
  return mesh ? std::string("type=\"mesh\" mesh=\"") + name + "\"" : "type=\"box\" size=\"" + size + "\"";
}

// A 2 x 2 m floor whose top is z = 0 and a free hull resting on it, each a box or a mesh (the
// cases take a box floor and a mesh hull unless they say otherwise), every geom with the given
// margin; no gravity.
MjModelPtr Model(bool floor_mesh, bool hull_mesh, double margin, const std::string& flags = "",
                 std::array<double, 3> size = {.2, .3, .1}) {
  std::string sz = std::to_string(size[0]) + " " + std::to_string(size[1]) + " " + std::to_string(size[2]);
  std::string xml =
      "<mujoco><compiler angle=\"radian\"/><option gravity=\"0 0 0\"><flag " + flags + "/></option>"
      "<default><geom margin=\"" + std::to_string(margin) + "\"/></default>"
      "<asset><mesh name=\"floor\" vertex=\"" + Corners(1, 1, .1) + "\"/>"
      "<mesh name=\"hull\" vertex=\"" + Corners(size[0], size[1], size[2]) + "\"/></asset>"
      "<worldbody><geom name=\"floor\" " + Geom(floor_mesh, "floor", "1 1 .1") + " pos=\"0 0 -.1\"/>"
      "<body name=\"hull\" pos=\"0 0 " + std::to_string(size[2]) + "\"><freejoint/>"
      "<geom name=\"hull\" " + Geom(hull_mesh, "hull", sz) + " mass=\"1\"/></body></worldbody></mujoco>";
  char error[1024];
  MjModelPtr m = LoadModelFromString(xml, error, sizeof(error));
  EXPECT_THAT(m.get(), NotNull()) << error;
  return m;
}

void SetQuat(mjData* d, double w, double x, double y, double z) {
  d->qpos[3] = w; d->qpos[4] = x; d->qpos[5] = y; d->qpos[6] = z;
}

TEST_F(ConvexManifoldTest, ParallelFacesAtSeparationAndPenetration) {
  for (bool floor_mesh : {true, false}) for (bool hull_mesh : {true, false}) {
    for (double margin : {.001, .02}) for (double gap : {-.0005, .0005}) {
      MjModelPtr m = Model(floor_mesh, hull_mesh, margin);
      MjDataPtr d = MakeData(m);
      d->qpos[2] += gap;
      mj_forward(m.get(), d.get());
      ASSERT_EQ(d->ncon, 4) << floor_mesh << hull_mesh << margin << gap;
      for (int i = 0; i < d->ncon; i++) {
        const mjContact& c = d->contact[i];
        EXPECT_NEAR(c.dist, gap, 3e-8);
        EXPECT_NEAR(c.pos[2], gap/2, 3e-8);
        double up = c.geom[0] == 0 ? 1 : -1;
        EXPECT_NEAR(c.frame[0], 0, 1e-8);
        EXPECT_NEAR(c.frame[1], 0, 1e-8);
        EXPECT_NEAR(c.frame[2], up, 1e-8);
        EXPECT_NEAR(std::abs(c.pos[0]), .2, 3e-8);
        EXPECT_NEAR(std::abs(c.pos[1]), .3, 3e-8);
      }
    }
  }
}

TEST_F(ConvexManifoldTest, NearTheMarginCoincidentWitnessesKeepTheFaceNormal) {
  // binary-exact extents put two tilted vertices within a few ulps of the margin, where their
  // inflated witnesses are too close together to normalize
  MjModelPtr m = Model(false, true, .015625, "", {.25, .25, .125});
  MjDataPtr d = MakeData(m);
  for (double angle : {.0001, .0005, .001}) {
    SetQuat(d.get(), std::cos(angle/2), 0, std::sin(angle/2), 0);
    d->qpos[2] = .03125 + .125*std::cos(angle) - .25*std::sin(angle);
    mj_forward(m.get(), d.get());
    ASSERT_EQ(d->ncon, 4) << angle;
    bool at_margin = false;
    for (int i = 0; i < d->ncon; i++) {
      at_margin |= std::abs(d->contact[i].dist - .03125) < 1e-14;
      EXPECT_NEAR(d->contact[i].frame[2], 1, 1e-8) << angle;
    }
    EXPECT_TRUE(at_margin) << angle;
  }
}

TEST_F(ConvexManifoldTest, OutsideTheMarginInventsNoContact) {
  for (bool floor_mesh : {false, true}) {
    MjModelPtr m = Model(floor_mesh, true, .01);
    MjDataPtr d = MakeData(m);
    d->qpos[2] += .021;
    mj_forward(m.get(), d.get());
    EXPECT_EQ(d->ncon, 0);
  }
}

TEST_F(ConvexManifoldTest, TranslatedAndYawedTheNormalPointsFromGeom0ToGeom1) {
  for (auto [floor_mesh, hull_mesh] : {std::pair{false, true}, {true, false}, {true, true}}) {
    MjModelPtr m = Model(floor_mesh, hull_mesh, .01);
    MjDataPtr d = MakeData(m);
    for (double yaw : {0.0, .1, .4, 1.1}) {
      d->qpos[0] = .15; d->qpos[1] = -.12; d->qpos[2] = .1005;
      SetQuat(d.get(), std::cos(yaw/2), 0, 0, std::sin(yaw/2));
      mj_forward(m.get(), d.get());
      ASSERT_EQ(d->ncon, 4) << yaw;
      for (int i = 0; i < d->ncon; i++) {
        const mjContact& c = d->contact[i];
        EXPECT_NEAR(c.dist, .0005, 3e-8);
        mjtNum between[3];
        mju_sub3(between, d->geom_xpos + 3*c.geom[1], d->geom_xpos + 3*c.geom[0]);
        EXPECT_GT(mju_dot3(c.frame, between), 0) << yaw;
      }
    }
  }
}

TEST_F(ConvexManifoldTest, TiltedFaceWitnessesHaveGeometricDepth) {
  MjModelPtr m = Model(false, true, .01);
  MjDataPtr d = MakeData(m);
  double angle = .0005;
  d->qpos[0] = 0; d->qpos[1] = 0; d->qpos[2] = .1005;
  SetQuat(d.get(), std::cos(angle/2), 0, std::sin(angle/2), 0);
  mj_forward(m.get(), d.get());
  ASSERT_EQ(d->ncon, 4);
  for (int i = 0; i < d->ncon; i++) {
    const mjContact& c = d->contact[i];
    // the two witness points from the midpoint and the signed distance
    mjtNum a[3], b[3];
    mju_addScl3(a, c.pos, c.frame, -.5*c.dist);
    mju_addScl3(b, c.pos, c.frame, .5*c.dist);
    if (c.geom[0] != 0) std::swap(a, b);
    EXPECT_NEAR(a[2], 0, 1e-8);
    mjtNum rel[3], local[3];  // the hull's witness in its body frame: on its face at -0.1
    mju_sub3(rel, b, d->xpos + 3);
    mju_mulMatTVec3(local, d->xmat + 9, rel);
    EXPECT_NEAR(local[2], -.1, 1e-7);
    EXPECT_LT(c.dist, .01);
  }
}

TEST_F(ConvexManifoldTest, AnUnalignedEdgeGivesOneFiniteContact) {
  MjModelPtr m = Model(false, true, .01);
  MjDataPtr d = MakeData(m);
  double angle = .4;
  d->qpos[2] = .1*std::cos(angle) + .3*std::sin(angle) - .0005;
  SetQuat(d.get(), std::cos(angle/2), std::sin(angle/2), 0, 0);
  mj_forward(m.get(), d.get());
  ASSERT_EQ(d->ncon, 1);
  for (int k = 0; k < 9; k++) EXPECT_TRUE(std::isfinite(d->contact[0].frame[k]));
  EXPECT_LT(d->contact[0].dist, 0);
}

TEST_F(ConvexManifoldTest, WithMulticcdDisabledItStaysOneContact) {
  MjModelPtr m = Model(false, true, .01, "multiccd=\"disable\"");
  MjDataPtr d = MakeData(m);
  d->qpos[2] += .0005;
  mj_forward(m.get(), d.get());
  EXPECT_EQ(d->ncon, 1);
}

TEST_F(ConvexManifoldTest, TheLibccdFallbackStaysAvailable) {
  MjModelPtr m = Model(false, true, .01, "nativeccd=\"disable\"");
  MjDataPtr d = MakeData(m);
  d->qpos[2] -= .0005;
  mj_forward(m.get(), d.get());
  EXPECT_GT(d->ncon, 0);
  EXPECT_LE(d->ncon, 5);
  for (int i = 0; i < d->ncon; i++) {
    for (int k = 0; k < 3; k++) EXPECT_TRUE(std::isfinite(d->contact[i].pos[k]));
  }
}

TEST_F(ConvexManifoldTest, TheReductionRespectsTheMeshCap) {
  MjModelPtr m = Model(true, true, .01, "", {.8, .8, .1});
  MjDataPtr d = MakeData(m);
  SetQuat(d.get(), std::cos(mjPI/8), 0, 0, std::sin(mjPI/8));
  d->qpos[2] += .0005;
  mj_forward(m.get(), d.get());
  ASSERT_EQ(d->ncon, 4);
  std::set<std::array<long long, 3>> points;
  for (int i = 0; i < d->ncon; i++) {
    EXPECT_NEAR(d->contact[i].dist, .0005, 3e-8);
    points.insert({std::llround(d->contact[i].pos[0]*1e8), std::llround(d->contact[i].pos[1]*1e8),
                   std::llround(d->contact[i].pos[2]*1e8)});
  }
  EXPECT_EQ(points.size(), 4);
}

TEST_F(ConvexManifoldTest, AZeroMarginKeepsNativeFaceClipping) {
  MjModelPtr m = Model(false, true, 0);
  MjDataPtr d = MakeData(m);
  d->qpos[2] -= .0005;
  mj_forward(m.get(), d.get());
  ASSERT_EQ(d->ncon, 4);
  for (int i = 0; i < d->ncon; i++) EXPECT_NEAR(d->contact[i].dist, -.0005, 3e-8);
}

}  // namespace
}  // namespace mujoco

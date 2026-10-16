// Copyright 2026 Feather contributors. Licensed under Apache-2.0.

// Tests for the fork's broadphase and midphase in engine/engine_collision_driver.c: the world-axis
// body boxes and split sweep, the node masks, the class-grouped body trees, the covered body pairs
// and the one-pair collisions. Each only skips work, so the contacts of a step must be exactly the
// ones a brute-force pass finds: every geom pair MuJoCo's filters admit, through its own collision
// function. The midphase's skips also turn off with vis.global.bvactive, whose contacts must match.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_collision_driver.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using BroadphaseTest = MujocoTest;

// A ground plane and two static walls (one a body of its own), then free bodies of one to three
// geoms in four collision classes; an explicit pair with its own margin, a body pair excluded, and
// a hinged child whose contacts with its parent the parent filter drops.
std::string ModelXml() {
  std::string xml =
      "<mujoco><option timestep=\"0.002\"/><default><geom margin=\"0.002\"/></default><worldbody>"
      "<geom name=\"ground\" type=\"plane\" size=\"0 0 1\"/>"
      "<geom name=\"wall\" type=\"box\" size=\"0.05 1 0.4\" pos=\"0.8 0 0.4\"/>"
      "<body name=\"static\" pos=\"-0.8 0 0.3\">"
      "<geom type=\"box\" size=\"0.05 1 0.3\"/><geom type=\"sphere\" size=\"0.1\" pos=\"0 0.5 0.4\"/>"
      "</body>";
  // classes: 1 collides with everything, 2 only with the static world, 4 only with 4 and the world
  const char* klass[] = {"contype=\"1\" conaffinity=\"1\"", "contype=\"2\" conaffinity=\"1\"",
                         "contype=\"4\" conaffinity=\"5\"", "contype=\"1\" conaffinity=\"3\""};
  const char* shape[] = {"type=\"sphere\" size=\"0.06\"", "type=\"capsule\" size=\"0.03 0.08\"",
                         "type=\"box\" size=\"0.05 0.07 0.04\"", "type=\"ellipsoid\" size=\"0.07 0.05 0.04\""};
  for (int b = 0; b < 14; b++) {
    xml += "<body name=\"b" + std::to_string(b) + "\"><freejoint/>";
    int ngeom = 1 + b % 3;
    for (int k = 0; k < ngeom; k++) {
      xml += "<geom name=\"g" + std::to_string(b) + "_" + std::to_string(k) + "\" " +
             shape[(b + k) % 4] + " " + klass[(b + 2*k) % 4] + " pos=\"" +
             std::to_string(0.07*k) + " 0 " + std::to_string(0.05*k) + "\"/>";
    }
    if (b == 3) {  // a hinged child, the parent filter's case
      xml += "<body name=\"child\" pos=\"0.12 0 0\"><joint type=\"hinge\" axis=\"0 0 1\"/>"
             "<geom name=\"childg\" type=\"capsule\" size=\"0.03 0.06\"/></body>";
    }
    xml += "</body>";
  }
  return xml +
         "</worldbody>"
         "<contact><pair geom1=\"g1_0\" geom2=\"g2_1\" margin=\"0.01\"/>"
         "<exclude body1=\"b4\" body2=\"b5\"/></contact></mujoco>";
}

class Random {
 public:
  explicit Random(uint64_t seed) : state_(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  double Uniform() {
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return (state_ >> 11) * (1.0 / 9007199254740992.0);
  }

 private:
  uint64_t state_;
};

// A contact keyed for comparison: the geom pair low id first, the normal pointing from it.
using Key = std::tuple<int, int, double, double, double, double, double, double, double>;

Key MakeKey(int g1, int g2, double dist, const mjtNum pos[3], const mjtNum normal[3]) {
  double sign = g1 < g2 ? 1 : -1;
  return Key(std::min(g1, g2), std::max(g1, g2), dist, pos[0], pos[1], pos[2],
             sign*normal[0], sign*normal[1], sign*normal[2]);
}

std::vector<Key> StepContacts(const mjModel* m, const mjData* d) {
  std::vector<Key> keys;
  for (int i = 0; i < d->ncon; i++) {
    const mjContact& c = d->contact[i];
    keys.push_back(MakeKey(c.geom[0], c.geom[1], c.dist, c.pos, c.frame));
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

// Every geom pair MuJoCo's filters admit, collided with its own collision function. The rules
// (filterBodyPair, filterCollisionPair, the explicit pairs' merge) for a model with no sleep, no
// gap, no flex and no contact filter callback.
std::vector<Key> BruteForce(const mjModel* m, mjData* d) {
  std::vector<Key> keys;
  auto collide = [&](int g1, int g2, mjtNum margin) {
    if (m->geom_type[g1] > m->geom_type[g2]) std::swap(g1, g2);
    mjfCollision func = mjCOLLISIONFUNC[m->geom_type[g1]][m->geom_type[g2]];
    if (!func) return;
    mjPreContact con[mjMAXCONPAIR];
    int n = func(m, d, con, g1, g2, margin);
    for (int i = 0; i < n; i++) keys.push_back(MakeKey(g1, g2, con[i].dist, con[i].pos, con[i].normal));
  };
  auto listed = [&](int g1, int g2) {
    for (int p = 0; p < m->npair; p++) {
      if ((m->pair_geom1[p] == g1 && m->pair_geom2[p] == g2) ||
          (m->pair_geom1[p] == g2 && m->pair_geom2[p] == g1)) return true;
    }
    return false;
  };
  auto excluded = [&](int b1, int b2) {
    int signature = (std::min(b1, b2) << 16) + std::max(b1, b2);
    for (int e = 0; e < m->nexclude; e++) {
      if (m->exclude_signature[e] == signature) return true;
    }
    return false;
  };
  for (int p = 0; p < m->npair; p++) {
    collide(m->pair_geom1[p], m->pair_geom2[p], m->pair_margin[p]);
  }
  for (int g1 = 0; g1 < m->ngeom; g1++) {
    for (int g2 = g1 + 1; g2 < m->ngeom; g2++) {
      int w1 = m->body_weldid[m->geom_bodyid[g1]], w2 = m->body_weldid[m->geom_bodyid[g2]];
      if (w1 == w2) continue;
      if (!m->body_dofnum[w1] && !m->body_dofnum[w2]) continue;
      int p1 = m->body_weldid[m->body_parentid[w1]], p2 = m->body_weldid[m->body_parentid[w2]];
      if (w1 && w2 && (w1 == p2 || w2 == p1)) continue;
      if (excluded(m->geom_bodyid[g1], m->geom_bodyid[g2])) continue;
      if (listed(g1, g2)) continue;
      if (!(m->geom_contype[g1] & m->geom_conaffinity[g2]) &&
          !(m->geom_contype[g2] & m->geom_conaffinity[g1])) continue;
      collide(g1, g2, m->geom_margin[g1] + m->geom_margin[g2]);  // a dynamic pair's margins add
    }
  }
  std::sort(keys.begin(), keys.end());
  return keys;
}

TEST_F(BroadphaseTest, AStepsContactsAreTheBruteForcePass) {
  char error[1024];
  MjModelPtr model = LoadModelFromString(ModelXml(), error, sizeof(error));
  ASSERT_THAT(model.get(), NotNull()) << error;
  mjModel* m = model.get();
  MjDataPtr data = MakeData(model);
  mjData* d = data.get();
  ASSERT_GT(m->npair, 0);
  ASSERT_GT(m->nexclude, 0);
  Random random(7);
  int touching = 0;
  for (int trial = 0; trial < 60; trial++) {
    // bodies dropped into a 1.2 m box over the ground, crowded enough to touch each other
    for (int j = 0; j < m->njnt; j++) {
      if (m->jnt_type[j] != mjJNT_FREE) {
        d->qpos[m->jnt_qposadr[j]] = 2*mjPI*random.Uniform();
        continue;
      }
      mjtNum* q = d->qpos + m->jnt_qposadr[j];
      q[0] = 0.6*(2*random.Uniform() - 1);
      q[1] = 0.6*(2*random.Uniform() - 1);
      q[2] = 0.05 + 0.4*random.Uniform();
      for (int k = 3; k < 7; k++) q[k] = 2*random.Uniform() - 1;
      mju_normalize4(q + 3);
    }
    mj_fwdPosition(m, d);
    std::vector<Key> step = StepContacts(m, d);
    std::vector<Key> brute = BruteForce(m, d);
    ASSERT_EQ(step.size(), brute.size()) << "trial " << trial;
    for (size_t i = 0; i < step.size(); i++) {
      SCOPED_TRACE(::testing::Message() << "geoms " << std::get<0>(step[i]) << " (type "
                   << m->geom_type[std::get<0>(step[i])] << ") and " << std::get<1>(step[i])
                   << " (type " << m->geom_type[std::get<1>(step[i])] << ")");
      // the same pair, depth and point; the normal to rounding, as the contact frame normalizes it
      EXPECT_EQ(std::get<0>(step[i]), std::get<0>(brute[i])) << "trial " << trial;
      EXPECT_EQ(std::get<1>(step[i]), std::get<1>(brute[i])) << "trial " << trial;
      EXPECT_EQ(std::get<2>(step[i]), std::get<2>(brute[i])) << "trial " << trial;
      EXPECT_EQ(std::get<3>(step[i]), std::get<3>(brute[i])) << "trial " << trial;
      EXPECT_EQ(std::get<4>(step[i]), std::get<4>(brute[i])) << "trial " << trial;
      EXPECT_EQ(std::get<5>(step[i]), std::get<5>(brute[i])) << "trial " << trial;
      EXPECT_NEAR(std::get<6>(step[i]), std::get<6>(brute[i]), 1e-12) << "trial " << trial;
      EXPECT_NEAR(std::get<7>(step[i]), std::get<7>(brute[i]), 1e-12) << "trial " << trial;
      EXPECT_NEAR(std::get<8>(step[i]), std::get<8>(brute[i]), 1e-12) << "trial " << trial;
    }
    touching += !step.empty();

    // the midphase's skips off (node masks, covered pairs): the same contacts in the same order
    std::vector<mjContact> fast(d->contact, d->contact + d->ncon);
    m->vis.global.bvactive = 1;
    mj_fwdPosition(m, d);
    m->vis.global.bvactive = 0;
    ASSERT_EQ(d->ncon, static_cast<int>(fast.size())) << "trial " << trial;
    for (int i = 0; i < d->ncon; i++) {
      EXPECT_EQ(d->contact[i].geom[0], fast[i].geom[0]) << "trial " << trial;
      EXPECT_EQ(d->contact[i].geom[1], fast[i].geom[1]) << "trial " << trial;
      EXPECT_EQ(d->contact[i].dist, fast[i].dist) << "trial " << trial;
    }
  }
  EXPECT_GT(touching, 50) << "the trials must be crowded enough to touch";
}

}  // namespace
}  // namespace mujoco

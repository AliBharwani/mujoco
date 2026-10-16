// Copyright 2026 Feather contributors. Licensed under Apache-2.0.

// Tests for engine/engine_collision_hfield.c, the collision table's heightfield x convex entry
// (mjc_SmoothHField): its depth against a dense brute-force oracle over five terrains, its parity
// with MuJoCo's plane functions on a flat field, its continuity while a geom slides across cells,
// and the prisms it leaves to upstream. The bounds are integration13's measurements with margin:
// they catch a regression, they do not define the method.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <mujoco/mujoco.h>
#include "src/engine/engine_collision_driver.h"
#include "test/fixture.h"

namespace mujoco {
namespace {

using ::testing::NotNull;
using SmoothHFieldTest = MujocoTest;

constexpr int kRows = 41, kCols = 41;
constexpr double kSx = 4, kSy = 4, kSz = 0.2, kBase = 0.1;
constexpr double kMargin = 0.002;

// the arena's ankle hull (mh1_L_Ankle_g0), as compiled
constexpr char kFoot[] =
    "0.022293 0.014166 -0.104285  -0.024120 0.014361 0.116657  -0.004171 -0.054125 0.015988 "
    "0.027970 0.043766 -0.006286  -0.021601 -0.042284 -0.064358  0.017802 -0.045030 0.089290 "
    "-0.037270 0.018475 -0.063451  0.040993 -0.014898 -0.057945  0.018749 0.046468 -0.060212 "
    "-0.026513 -0.013464 -0.104622  -0.017892 0.035857 -0.023218  0.042274 -0.001080 -0.011786 "
    "0.010851 -0.005746 0.103722  0.001919 0.037057 0.095833  -0.002124 -0.056352 0.055350 "
    "-0.028681 0.010104 0.082173  -0.038645 -0.013498 -0.075003  -0.009195 -0.030894 0.097479 "
    "-0.007991 0.012536 -0.099828  -0.007865 -0.032533 -0.088526  -0.028151 0.035893 0.095725 "
    "0.041175 0.021006 0.006349  0.010068 -0.008773 -0.100472  -0.035688 0.011513 -0.087712 "
    "-0.003808 -0.050099 0.078767  -0.016527 -0.006908 0.112787  0.024375 0.051610 -0.027420 "
    "-0.031184 -0.030417 -0.087642  0.042282 -0.004023 -0.033688  -0.009521 -0.048387 0.036443 "
    "-0.030935 -0.025603 -0.054316  0.037720 -0.005013 -0.076601  0.013354 -0.053756 0.069465 "
    "0.015902 -0.026884 0.095435  0.001193 -0.048904 0.097407  0.008070 -0.054595 0.033834 "
    "-0.041202 -0.005692 -0.092320  -0.007978 -0.013371 -0.105672  -0.024951 0.004973 -0.100845 "
    "-0.015368 0.040216 0.107582  0.033766 0.035920 0.007090  -0.019051 -0.027517 -0.099283 "
    "0.016037 0.050564 -0.044754  -0.044956 0.005940 -0.069695  -0.028883 0.028475 0.111104 "
    "0.042786 0.006112 0.002186  -0.039877 -0.006812 -0.061202  -0.017816 -0.040950 -0.080367 "
    "0.022664 0.038827 -0.072566  0.008805 -0.056459 0.085591  -0.032675 -0.025600 -0.068924 "
    "-0.027209 0.011414 0.096746  -0.035967 -0.018627 -0.095193  -0.013458 -0.018774 0.103045 "
    "-0.002920 -0.001000 -0.106465  -0.015102 0.041282 0.094518  -0.000547 0.034862 0.108354 "
    "0.018458 0.023678 -0.096691  -0.000654 -0.056436 0.068380  -0.009411 -0.042766 -0.066406 "
    "-0.044071 0.002520 -0.083658  -0.031240 0.022010 0.092536  -0.014311 0.002199 -0.105048 "
    "-0.004205 -0.054903 0.028538";

enum Kind { kSphere, kCapsule, kBox, kBlade, kFootMesh, kRock, kEllipsoid, kCylinder, kKinds };
constexpr const char* kName[kKinds] = {"sphere", "capsule", "box", "blade", "foot", "rock",
                                       "ellipsoid", "cylinder"};
constexpr const char* kShape[kKinds] = {
    R"(type="sphere" size="0.05")",
    R"(type="capsule" size="0.03 0.12")",
    R"(type="box" size="0.12 0.08 0.05")",
    R"(type="box" size="0.16 0.012 0.003")",
    R"(type="mesh" mesh="foot")",
    R"(type="mesh" mesh="rock")",
    R"(type="ellipsoid" size="0.09 0.05 0.03")",
    R"(type="cylinder" size="0.05 0.04")",
};
// the shapes the smooth collider handles
constexpr Kind kSmooth[] = {kSphere, kCapsule, kBox, kBlade, kFootMesh, kRock};

// One free body per kind over the field, and a plane at the field's flat height for the parity
// test, which collides only when called directly.
std::string ModelXml() {
  std::string xml =
      "<mujoco><option timestep=\"0.0022222222\"/>"
      "<default><geom margin=\"0.001\"/></default><asset>"
      "<hfield name=\"h\" nrow=\"" + std::to_string(kRows) + "\" ncol=\"" + std::to_string(kCols) +
      "\" size=\"4 4 0.2 0.1\"/>"
      "<mesh name=\"foot\" vertex=\"" + std::string(kFoot) + "\"/>"
      "<mesh name=\"rock\" vertex=\"0.07 0 0.02  -0.05 0.05 0.03  -0.04 -0.06 0.01  0 0 -0.05  "
      "0.02 0.06 0.05  0.03 -0.03 0.06  -0.07 0 -0.01  0.05 0.04 -0.03\"/></asset><worldbody>"
      "<geom name=\"terrain\" type=\"hfield\" hfield=\"h\"/>"
      "<geom name=\"floor\" type=\"plane\" size=\"0 0 1\" pos=\"0 0 0.1\" contype=\"0\" "
      "conaffinity=\"0\"/>";
  for (int k = 0; k < kKinds; k++) {
    xml += "<body name=\"" + std::string(kName[k]) + "\" pos=\"" + std::to_string(-3 + 0.8*k) +
           " 0 0.5\"><freejoint/><geom name=\"g_" + kName[k] + "\" " + kShape[k] + "/></body>";
  }
  return xml + "</worldbody></mujoco>";
}

// a fixed-seed generator, so every platform draws the same poses and terrains
class Random {
 public:
  explicit Random(uint64_t seed) : state_(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  double Uniform() {  // [0, 1)
    state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
    return (state_ >> 11) * (1.0 / 9007199254740992.0);
  }
  void Quat(mjtNum q[4]) {  // uniform over rotations
    double u1 = Uniform(), u2 = 2*mjPI*Uniform(), u3 = 2*mjPI*Uniform();
    q[0] = std::sqrt(1 - u1)*std::sin(u2); q[1] = std::sqrt(1 - u1)*std::cos(u2);
    q[2] = std::sqrt(u1)*std::sin(u3); q[3] = std::sqrt(u1)*std::cos(u3);
  }

 private:
  uint64_t state_;
};

enum Terrain { kFlat, kSlope, kDimples, kRidges, kRough, kTerrains };

void WriteTerrain(mjModel* m, Terrain kind, uint64_t seed) {
  Random random(seed);
  double cx[6], cy[6];
  for (int i = 0; i < 6; i++) {
    cx[i] = -1.2 + 2.4*random.Uniform();
    cy[i] = -1.2 + 2.4*random.Uniform();
  }
  float* data = m->hfield_data + m->hfield_adr[0];
  for (int r = 0; r < kRows; r++) {
    for (int c = 0; c < kCols; c++) {
      double x = -kSx + 2*kSx*c/(kCols - 1), y = -kSy + 2*kSy*r/(kRows - 1), h = 0.1;
      switch (kind) {
        case kSlope:
          h += 0.06*x/kSx + 0.04*y/kSy;
          break;
        case kDimples:
          for (int i = 0; i < 6; i++) {
            double dx = x - cx[i], dy = y - cy[i];
            h -= 0.015*std::exp(-(dx*dx + dy*dy)/(2*0.12*0.12));
          }
          h += 0.002*(random.Uniform() - 0.5);
          break;
        case kRidges: {
          // a crest on lattice column x = 0, and one off the lattice along y = 0.63 x + 0.1
          double a = std::max(0.0, 1 - std::abs(x)/0.2);
          double b = std::max(0.0, 1 - std::abs(y - 0.63*x - 0.1)/0.25);
          h += 0.02*std::max(a, b);
          break;
        }
        case kRough:
          h += 0.01*(random.Uniform() - 0.5);
          break;
        default:
          break;
      }
      data[r*kCols + c] = static_cast<float>(h/kSz);
    }
  }
}

class Scene {
 public:
  Scene() {
    char error[1024];
    model_ = LoadModelFromString(ModelXml(), error, sizeof(error));
    EXPECT_THAT(model_.get(), NotNull()) << error;
    data_ = MakeData(model_);
  }
  mjModel* m() { return model_.get(); }
  mjData* d() { return data_.get(); }
  int Geom(Kind k) { return mj_name2id(m(), mjOBJ_GEOM, (std::string("g_") + kName[k]).c_str()); }

  void Place(Kind k, const mjtNum pos[3], const mjtNum quat[4]) {
    int adr = m()->jnt_qposadr[m()->body_jntadr[mj_name2id(m(), mjOBJ_BODY, kName[k])]];
    mju_copy3(d()->qpos + adr, pos);
    mju_copy4(d()->qpos + adr + 3, quat);
    mj_kinematics(m(), d());
  }

  double TerrainZ(double x, double y) {
    mjtNum p[3] = {x, y, 5}, v[3] = {0, 0, -1};
    mjtNum t = mj_rayHfield(m(), d(), 0, p, v, nullptr);
    return t < 0 ? -1e9 : 5 - t;
  }

  double Bottom(int g, double x, double y) {
    mjtNum p[3] = {x, y, -5}, v[3] = {0, 0, 1};
    mjtNum t = m()->geom_type[g] == mjGEOM_MESH
                   ? mj_rayMesh(m(), d(), g, p, v, nullptr)
                   : mju_rayGeom(d()->geom_xpos + 3*g, d()->geom_xmat + 9*g, m()->geom_size + 3*g,
                                 p, v, m()->geom_type[g], nullptr);
    return t < 0 ? 1e9 : -5 + t;
  }

  // the geom over (x, y) with its lowest point `clearance` above the terrain below it
  void RestOn(Kind k, double x, double y, const mjtNum q[4], double clearance) {
    mjtNum pos[3] = {x, y, 1};
    Place(k, pos, q);
    int g = Geom(k);
    double low = 1e9, radius = m()->geom_rbound[g];
    for (double a = -radius; a <= radius; a += radius/12) {
      for (double b = -radius; b <= radius; b += radius/12) {
        double z = Bottom(g, x + a, y + b);
        if (z < 1e8) low = std::min(low, z - TerrainZ(x + a, y + b));
      }
    }
    pos[2] = 1 - low + clearance;
    Place(k, pos, q);
  }

  // the smallest vertical gap between the geom and the terrain: its lower surface sampled every
  // `step` metres, and a box's or mesh's vertices exactly
  double OracleGap(int g, double step) {
    double gap = 1e9, radius = m()->geom_rbound[g];
    const mjtNum* c = d()->geom_xpos + 3*g;
    for (double x = c[0] - radius; x <= c[0] + radius; x += step) {
      for (double y = c[1] - radius; y <= c[1] + radius; y += step) {
        double z = Bottom(g, x, y);
        if (z < 1e8) gap = std::min(gap, z - TerrainZ(x, y));
      }
    }
    int type = m()->geom_type[g];
    int nvert = type == mjGEOM_BOX ? 8 : type == mjGEOM_MESH ? m()->mesh_vertnum[m()->geom_dataid[g]] : 0;
    for (int i = 0; i < nvert; i++) {
      mjtNum local[3], world[3];
      if (type == mjGEOM_BOX) {
        for (int j = 0; j < 3; j++) {
          local[j] = (i & (1 << j)) ? m()->geom_size[3*g + j] : -m()->geom_size[3*g + j];
        }
      } else {
        const float* v = m()->mesh_vert + 3*(m()->mesh_vertadr[m()->geom_dataid[g]] + i);
        local[0] = v[0]; local[1] = v[1]; local[2] = v[2];
      }
      mju_mulMatVec3(world, d()->geom_xmat + 9*g, local);
      mju_addTo3(world, c);
      gap = std::min(gap, world[2] - TerrainZ(world[0], world[1]));
    }
    return gap;
  }

  int Collide(int g, mjPreContact* con) {
    return mjCOLLISIONFUNC[mjGEOM_HFIELD][m()->geom_type[g]](m(), d(), con, 0, g, kMargin);
  }

 private:
  MjModelPtr model_;
  MjDataPtr data_;
};

int Deepest(const mjPreContact* con, int n) {
  int deepest = 0;
  for (int i = 1; i < n; i++) {
    if (con[i].dist < con[deepest].dist) deepest = i;
  }
  return deepest;
}

TEST_F(SmoothHFieldTest, DepthMatchesABruteForceOracle) {
  Scene scene;
  for (int t = 0; t < kTerrains; t++) {
    WriteTerrain(scene.m(), static_cast<Terrain>(t), 17 + t);
    for (Kind k : kSmooth) {
      int g = scene.Geom(k), tested = 0;
      Random random(1000*t + k);
      for (int i = 0; i < 24; i++) {
        mjtNum q[4] = {1, 0, 0, 0};
        if (k == kBlade) {  // a blade lies flat
          double yaw = 2*mjPI*random.Uniform();
          q[0] = std::cos(yaw/2);
          q[3] = std::sin(yaw/2);
        } else {
          random.Quat(q);
        }
        double x = -1 + 2*random.Uniform(), y = -1 + 2*random.Uniform();
        if (t == kRidges && i % 2) x = 0.3*(random.Uniform() - 0.5);  // across the lattice crest
        scene.RestOn(k, x, y, q, -0.006 + 0.007*random.Uniform());
        double oracle = scene.OracleGap(g, 0.001);
        if (oracle > kMargin) continue;
        tested++;
        mjPreContact con[mjMAXCONPAIR];
        int n = scene.Collide(g, con);
        ASSERT_LE(n, 4) << kName[k] << " on terrain " << t;
        double gap = kMargin;  // nothing found within the margin
        for (int j = 0; j < n; j++) {
          ASSERT_TRUE(std::isfinite(con[j].dist) && std::isfinite(con[j].normal[2]));
          ASSERT_GT(con[j].normal[2], 0) << kName[k] << " on terrain " << t;
          gap = std::min(gap, con[j].dist/con[j].normal[2]);
        }
        EXPECT_LT(gap - oracle, 0.0003) << kName[k] << " on terrain " << t << ": shallower";
        EXPECT_LT(oracle - gap, 0.00035) << kName[k] << " on terrain " << t << ": deeper";
      }
      EXPECT_GT(tested, 12) << kName[k] << " on terrain " << t;
    }
  }
}

TEST_F(SmoothHFieldTest, OnAFlatFieldItIsThePlaneFunctions) {
  Scene scene;
  WriteTerrain(scene.m(), kFlat, 1);  // at the plane's height
  int floor = mj_name2id(scene.m(), mjOBJ_GEOM, "floor");
  for (Kind k : kSmooth) {
    int g = scene.Geom(k);
    Random random(77 + k);
    for (int i = 0; i < 40; i++) {
      mjtNum q[4];
      random.Quat(q);
      double x = -1 + 2*random.Uniform(), y = -1 + 2*random.Uniform();
      scene.RestOn(k, x, y, q, -0.004 + 0.005*random.Uniform());
      mjPreContact a[mjMAXCONPAIR], b[mjMAXCONPAIR];
      int na = scene.Collide(g, a);
      int nb = mjCOLLISIONFUNC[mjGEOM_PLANE][scene.m()->geom_type[g]](scene.m(), scene.d(), b,
                                                                        floor, g, kMargin);
      ASSERT_EQ(na > 0, nb > 0) << kName[k];
      if (!na) continue;
      // the deepest contact is the plane function's
      const mjPreContact& p = a[Deepest(a, na)];
      const mjPreContact& r = b[Deepest(b, nb)];
      EXPECT_NEAR(p.dist, r.dist, 1e-12) << kName[k];
      EXPECT_LT(mju_dist3(p.pos, r.pos), 1e-12) << kName[k];
      mjtNum dn[3];
      mju_sub3(dn, p.normal, r.normal);
      EXPECT_LT(mju_norm3(dn), 1e-12) << kName[k];
      // and for a primitive the whole set is (a hull's support polygon is chosen differently)
      if (scene.m()->geom_type[g] == mjGEOM_MESH) continue;
      ASSERT_EQ(na, nb) << kName[k];
      for (int j = 0; j < nb; j++) {
        bool found = false;
        for (int l = 0; l < na && !found; l++) {
          found = mju_dist3(a[l].pos, b[j].pos) < 1e-9 && std::abs(a[l].dist - b[j].dist) < 1e-9;
        }
        EXPECT_TRUE(found) << kName[k] << ": the plane's contact " << j << " is missing";
      }
    }
  }
}

TEST_F(SmoothHFieldTest, ASlideAcrossCellsIsContinuous) {
  Scene scene;
  const Kind kinds[] = {kBox, kFootMesh, kBlade, kCapsule, kSphere};
  for (int t = kSlope; t < kTerrains; t++) {
    WriteTerrain(scene.m(), static_cast<Terrain>(t), 5);
    for (Kind k : kinds) {
      int g = scene.Geom(k), previous = -1;
      mjtNum q[4] = {std::cos(0.2), 0, 0, std::sin(0.2)}, normal[3] = {0, 0, 0};
      double dist = 0, turn = 0, jump = 0;
      for (int s = 0; s <= 600; s++) {  // 0.3 m in 0.5 mm steps
        scene.RestOn(k, -0.3 + 0.0005*s, 0.07 + 0.3*0.0005*s, q, -0.002);
        mjPreContact con[mjMAXCONPAIR];
        int n = scene.Collide(g, con);
        if (!n) {
          previous = 0;
          continue;
        }
        const mjPreContact& c = con[Deepest(con, n)];
        if (previous > 0) {
          turn = std::max(turn, std::acos(mju_clip(mju_dot3(normal, c.normal), -1, 1))*180/mjPI);
          jump = std::max(jump, std::abs(c.dist - dist));
        }
        mju_copy3(normal, c.normal);
        dist = c.dist;
        previous = n;
      }
      EXPECT_LT(turn, 2.0) << kName[k] << " on terrain " << t << ": degrees per step";
      EXPECT_LT(jump, 0.00035) << kName[k] << " on terrain " << t << ": metres per step";
    }
  }
}

// The collider leaves cylinders and ellipsoids to upstream's prisms (mjc_ConvexHField, which the
// library does not export to compare against): one sunk 3 mm into rough ground still finds it.
TEST_F(SmoothHFieldTest, CylindersAndEllipsoidsStillCollideOnThePrisms) {
  Scene scene;
  WriteTerrain(scene.m(), kRough, 3);
  for (Kind k : {kEllipsoid, kCylinder}) {
    int g = scene.Geom(k);
    Random random(5 + k);
    for (int i = 0; i < 20; i++) {
      mjtNum q[4];
      random.Quat(q);
      scene.RestOn(k, -1 + 2*random.Uniform(), -1 + 2*random.Uniform(), q, -0.003);
      mjPreContact con[mjMAXCONPAIR];
      int n = scene.Collide(g, con);
      ASSERT_GT(n, 0) << kName[k];
      for (int j = 0; j < n; j++) ASSERT_TRUE(std::isfinite(con[j].dist)) << kName[k];
      EXPECT_LT(con[Deepest(con, n)].dist, 0) << kName[k];
    }
  }
}

}  // namespace
}  // namespace mujoco

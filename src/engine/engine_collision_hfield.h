// Copyright 2026 Feather contributors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// https://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

#ifndef MUJOCO_SRC_ENGINE_ENGINE_COLLISION_HFIELD_H_
#define MUJOCO_SRC_ENGINE_ENGINE_COLLISION_HFIELD_H_

#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>

#ifdef __cplusplus
extern "C" {
#endif

// The collision table's heightfield x convex entry: smooth-normal, feature-anchored contacts
// (at most four per pair) for spheres, capsules, boxes and polygon meshes over the field's
// interior; upstream mjc_ConvexHField for the side walls, the underside and other geoms.
int mjc_SmoothHField(const mjModel* m, mjData* d, mjPreContact* con,
                     int hgeom, int g, mjtNum margin);

// A geom's bounding box in a heightfield geom's frame (the smooth collider's footprint).
void mjc_hfieldGeomBox(const mjModel* m, const mjData* d, int hgeom, int g,
                       mjtNum lo[3], mjtNum hi[3]);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_COLLISION_HFIELD_H_

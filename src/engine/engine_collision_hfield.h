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

// Refresh the min/max pyramid for an inclusive rectangle of edited samples.
// Leaves cover 2x2 cells (3x3 samples). Levels are packed at hfield_adr;
// their total size is at most nrow*ncol, including rectangular/odd grids.
void mjc_updateHFieldMip(mjModel* m, int id, int rmin, int rmax, int cmin, int cmax);

// Top-surface contacts for a box/convex hull inside a heightfield's footprint.
// Returns -1 when the regular solid-prism collider must handle the pair instead.
int mjc_PolyHField(const mjModel* m, mjData* d, mjPreContact* con,
                  int hgeom, int pgeom, mjtNum margin);

// Smooth-normal, feature-anchored contacts for any convex primitive or polygon mesh
// over the heightfield's interior (at most four per pair). Returns -1 when another
// collider must handle the pair: unsupported types, meshes without polygons, or geoms
// reaching the field's side walls or underside.
int mjc_SmoothHField(const mjModel* m, mjData* d, mjPreContact* con,
                     int hgeom, int g, mjtNum margin);

#ifdef __cplusplus
}
#endif

#endif  // MUJOCO_SRC_ENGINE_ENGINE_COLLISION_HFIELD_H_

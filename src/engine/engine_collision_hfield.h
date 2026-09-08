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

// Top-surface contacts for a box/convex hull inside a heightfield's footprint.
// Returns -1 when the regular solid-prism collider must handle the pair instead.
int mjc_PolyHField(const mjModel* m, mjData* d, mjPreContact* con,
                  int hgeom, int pgeom, mjtNum margin);

#endif  // MUJOCO_SRC_ENGINE_ENGINE_COLLISION_HFIELD_H_

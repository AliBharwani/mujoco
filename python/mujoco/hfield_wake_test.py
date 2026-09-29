# Copyright 2026 Feather contributors. Licensed under Apache-2.0.
"""Regional wakes must respect actual edits, hfield frames and island ownership."""
import ctypes
from pathlib import Path
import unittest

import mujoco as mj
import numpy as np


def api():
  lib = ctypes.CDLL(str(Path(mj.__file__).parent/f'libmujoco.so.{mj.mj_versionString()}'))
  lib.mjf_writeHfieldRows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
  lib.mjf_wakeHfieldRegion.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                     ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
  return lib


def model(rotated=False, shared=False, coupled=False):
  # Translated and 90-degree yawed instance, with bodies authored in world space.
  floor = 'pos="10 20 0" quat=".7071067811865476 0 0 .7071067811865476"' if rotated else ''
  bodies = ''
  for i, x in enumerate((-2, 0, 2)):
    pos = f'10 {20+x} .1' if rotated else f'{x} 0 .1'
    bodies += f'<body name="rack{i}" pos="{pos}"><freejoint/>' \
              '<geom type="box" size=".15 .15 .1" mass="2"/></body>'
  if shared:
    bodies += '<geom name="second" type="hfield" hfield="h" pos="0 10 0"/>' \
              '<body name="instance_rack" pos="-2 10 .1"><freejoint/>' \
              '<geom type="box" size=".15 .15 .1" mass="2"/></body>'
  bodies += '<body name="humanoid" pos="0 3 .1" sleep="never"><freejoint/>' \
            '<geom type="box" size=".15 .15 .1" mass="2"/></body>'
  equality = '<equality><weld body1="rack0" body2="rack1"/></equality>' if coupled else ''
  return mj.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".0022222222222222222" integrator="implicitfast">
      <flag sleep="enable"/></option>
    <asset><hfield name="h" nrow="81" ncol="81" size="4 4 1 .1"/></asset>
    <worldbody><geom name="sand" type="hfield" hfield="h" {floor}/>{bodies}</worldbody>
    {equality}</mujoco>''')


class HfieldWakeTest(unittest.TestCase):
  def setUp(self):
    self.lib = api()

  def settled(self, **kwargs):
    m = model(**kwargs); d = mj.MjData(m)
    if kwargs.get('rotated'):
      # Keep the unrelated never-sleep body on the same transformed field.
      adr = m.jnt_qposadr[m.body_jntadr[m.body('humanoid').id]]
      d.qpos[adr:adr+3] = [7, 20, .1]
    for _ in range(30):
      mj.mj_step(m, d, nstep=100)
      if np.all(d.tree_asleep[:3] >= 0): break
    self.assertTrue(np.all(d.tree_asleep[:3] >= 0), d.tree_asleep)
    return m, d

  def write(self, m, d, columns, value=.001):
    row = m.hfield_data.reshape(81, 81)[40].copy()
    row[columns] = value
    return self.lib.mjf_writeHfieldRows(m._address, d._address, 0, 40, 1,
                                       row.ctypes.data, row.size)

  def test_one_print_wakes_one_rack_and_keeps_never_sleep_body_awake(self):
    m, d = self.settled()
    before = d.qpos.copy(), d.qvel.copy()
    self.assertEqual(self.write(m, d, [20]), 1)
    self.assertTrue(d.tree_asleep[0] < 0)
    self.assertTrue(np.all(d.tree_asleep[1:3] >= 0))
    self.assertTrue(d.tree_asleep[3] < 0)
    np.testing.assert_array_equal(d.qpos, before[0])
    np.testing.assert_array_equal(d.qvel, before[1])
    mj.mj_step(m, d)
    self.assertTrue(np.all(d.tree_asleep[1:3] >= 0))

  def test_disjoint_edits_do_not_wake_rack_between_them(self):
    m, d = self.settled()
    self.write(m, d, [20, 60])
    self.assertLess(d.tree_asleep[0], 0)
    self.assertGreaterEqual(d.tree_asleep[1], 0)
    self.assertLess(d.tree_asleep[2], 0)

  def test_identical_invalid_and_far_writes_do_not_wake(self):
    m, d = self.settled()
    before = d.tree_asleep.copy()
    self.assertEqual(self.write(m, d, [20], 0), 0)
    self.assertEqual(self.write(m, d, [20], np.nan), -1)
    self.assertEqual(self.write(m, d, [4]), 1)
    np.testing.assert_array_equal(d.tree_asleep, before)

  def test_adjacent_cell_changes_are_included(self):
    m, d = self.settled()
    # rack0 reaches x=-1.85; vertex -1.8 affects the cell beginning at -1.9.
    self.write(m, d, [22])
    self.assertLess(d.tree_asleep[0], 0)
    self.assertTrue(np.all(d.tree_asleep[1:3] >= 0))

  def test_rotated_translated_heightfield(self):
    m, d = self.settled(rotated=True)
    self.write(m, d, [20])
    self.assertLess(d.tree_asleep[0], 0)
    self.assertTrue(np.all(d.tree_asleep[1:3] >= 0))

  def test_contact_and_override_margins_expand_the_wake(self):
    for override in (False, True):
      m, d = self.settled()
      if override:
        m.opt.enableflags |= int(mj.mjtEnableBit.mjENBL_OVERRIDE)
        m.opt.o_margin = .3
      else:
        m.geom_margin[m.body_geomadr[m.body('rack0').id]] = .3
      self.write(m, d, [24])  # outside the raw AABB, inside its expanded margin
      self.assertLess(d.tree_asleep[0], 0)
      self.assertTrue(np.all(d.tree_asleep[1:3] >= 0))

  def test_both_geom_margins_are_added_as_in_312_collision_driver(self):
    m, d = self.settled()
    m.geom_margin[m.body_geomadr[m.body('rack0').id]] = .1
    m.geom_margin[m.geom('sand').id] = .1
    self.write(m, d, [24])  # requires their .2 sum, not either .1 alone
    self.assertLess(d.tree_asleep[0], 0)
    self.assertTrue(np.all(d.tree_asleep[1:3] >= 0))

  def test_every_instance_of_shared_height_asset_is_considered(self):
    m, d = self.settled(shared=True)
    self.assertGreaterEqual(d.tree_asleep[3], 0)
    self.write(m, d, [20])
    self.assertLess(d.tree_asleep[0], 0)
    self.assertLess(d.tree_asleep[3], 0)
    self.assertTrue(np.all(d.tree_asleep[1:3] >= 0))

  def test_connected_island_wakes_together(self):
    m, d = self.settled(coupled=True)
    self.write(m, d, [20])
    self.assertTrue(np.all(d.tree_asleep[:2] < 0))
    self.assertGreaterEqual(d.tree_asleep[2], 0)

  def test_other_data_can_be_woken_after_shared_model_write(self):
    m, first = self.settled()
    other = mj.MjData(m)
    mj.mj_copyData(other, m, first)
    self.write(m, first, [20])
    self.assertTrue(np.all(other.tree_asleep[:3] >= 0))
    self.assertEqual(self.lib.mjf_wakeHfieldRegion(m._address, other._address, 0,
                                                40, 1, 20, 1), 1)
    self.assertLess(other.tree_asleep[0], 0)
    self.assertTrue(np.all(other.tree_asleep[1:3] >= 0))
    self.assertEqual(self.lib.mjf_wakeHfieldRegion(m._address, other._address, 0,
                                                -1, 1, 20, 1), -1)

  def test_upstream_contact_wake_still_operates(self):
    m, d = self.settled()
    # Move the never-sleep body into the first rack; upstream contact wake owns this.
    body = m.body('humanoid').id
    adr = m.jnt_qposadr[m.body_jntadr[body]]
    d.qpos[adr:adr+3] = [-1.8, 0, .1]
    mj.mj_step(m, d)
    self.assertLess(d.tree_asleep[0], 0)
    self.assertTrue(np.all(d.tree_asleep[1:3] >= 0))


if __name__ == '__main__':
  unittest.main()

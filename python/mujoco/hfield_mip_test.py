# Copyright 2026 Feather contributors. Licensed under Apache-2.0.
"""Cache lifetime, conservative culling and incremental hfield writes."""
import copy
import ctypes
import itertools
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np


def api():
  lib = ctypes.CDLL(str(Path(mujoco.__file__).parent /
                        f'libmujoco.so.{mujoco.mj_versionString()}'))
  lib.mjf_enableHfieldMip.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
  lib.mjf_writeHfieldRows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                    ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
  return lib


def make_model(nr=65, nc=67, kind='mesh'):
  vertices = ' '.join(str(v) for xyz in itertools.product(
      (-.31, .31), (-.72, .72), (-.0795, .0795)) for v in xyz)
  geom = 'type="box" size=".31 .72 .0795"' if kind == 'box' else 'type="mesh" mesh="hull"'
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".0022222222222222222" integrator="implicitfast"/>
    <asset><hfield name="h" nrow="{nr}" ncol="{nc}" size="2 3 .3 .1"/>
      <mesh name="hull" vertex="{vertices}"/></asset>
    <worldbody><geom type="hfield" hfield="h" quat=".99 .05 .02 .01"/>
      <body><freejoint/><geom {geom} mass="1" margin=".001"/></body>
    </worldbody></mujoco>''')


def state(d):
  return [x.copy() for x in (d.qpos, d.qvel, d.qacc, d.qacc_warmstart,
                            d.efc_force, d.contact.pos, d.contact.frame, d.contact.dist)]


class HfieldMipTest(unittest.TestCase):
  def setUp(self):
    self.lib = api()

  def test_raw_writes_remain_uncached_by_default(self):
    m = make_model()
    self.assertEqual(m.hfield_mipvalid[0], 0)
    d = mujoco.MjData(m)
    d.qpos[2] = .2
    m.hfield_data[:] = 0
    mujoco.mj_forward(m, d)
    self.assertEqual(d.ncon, 0)
    m.hfield_data[:] = .5
    mujoco.mj_forward(m, d)
    self.assertGreater(d.ncon, 0)

  def test_random_geometry_and_margins_match_uncached_exactly(self):
    rng = np.random.default_rng(9092026)
    for kind, dims in itertools.product(('box', 'mesh'), ((2, 67), (33, 49), (64, 66), (129, 151))):
      m = make_model(*dims, kind=kind)
      m.hfield_data[:] = rng.uniform(.1, .9, m.nhfielddata)
      self.assertEqual(self.lib.mjf_enableHfieldMip(m._address, 0, 1), 0)
      ref = copy.copy(m)
      self.lib.mjf_enableHfieldMip(ref._address, 0, 0)
      d, r = mujoco.MjData(m), mujoco.MjData(ref)
      for i in range(100):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        d.qpos[:3] = rng.uniform([-.8, -1, -.1], [.8, 1, .7])
        d.qpos[3:] = q
        r.qpos[:] = d.qpos
        margin = (0, .001, .03, .2)[i % 4]
        m.geom_margin[1] = ref.geom_margin[1] = margin
        mujoco.mj_forward(m, d); mujoco.mj_forward(ref, r)
        for got, expected in zip(state(d), state(r)):
          np.testing.assert_array_equal(got, expected, err_msg=f'{kind} {dims} {i}')

  def test_partial_rows_match_full_refit_including_seams_and_shrinking_max(self):
    rng = np.random.default_rng(123)
    for nr, nc in ((2, 67), (65, 2), (65, 67), (64, 66)):
      m = make_model(nr, nc)
      m.hfield_data[:] = rng.random(m.nhfielddata)
      self.lib.mjf_enableHfieldMip(m._address, 0, 1)
      d = mujoco.MjData(m)
      for row in (0, 1, nr//2, nr-2, nr-1):
        for value in (0, 1, .2):
          data = np.full(nc, value, dtype=np.float32)
          expected = m.hfield_data.copy().reshape(nr, nc)
          expected[row] = data
          self.lib.mjf_writeHfieldRows(m._address, d._address, 0, row, 1, data.ctypes.data, nc)
          np.testing.assert_array_equal(m.hfield_data.reshape(nr, nc), expected)
          incremental = (m.hfield_mipmin.copy(), m.hfield_mipmax.copy())
          self.lib.mjf_enableHfieldMip(m._address, 0, 1)
          np.testing.assert_array_equal(m.hfield_mipmin, incremental[0])
          np.testing.assert_array_equal(m.hfield_mipmax, incremental[1])

  def test_invalid_writes_are_atomic_and_disabled_cache_can_be_refreshed(self):
    m = make_model(); d = mujoco.MjData(m)
    self.lib.mjf_enableHfieldMip(m._address, 0, 1)
    before = (m.hfield_data.copy(), m.hfield_mipmax.copy())
    data = np.ones(67, np.float32); data[-1] = np.nan
    for row, count in ((0, 67), (-1, 67), (65, 67), (0, 66)):
      self.assertEqual(self.lib.mjf_writeHfieldRows(m._address, d._address, 0, row, 1,
                                                 data.ctypes.data, count), -1)
      np.testing.assert_array_equal(m.hfield_data, before[0])
      np.testing.assert_array_equal(m.hfield_mipmax, before[1])
    self.lib.mjf_enableHfieldMip(m._address, 0, 0)
    m.hfield_data[:] = .5
    self.assertEqual(self.lib.mjf_enableHfieldMip(m._address, 0, 1), 0)
    self.assertEqual(m.hfield_mipmax[0], .5)
    m.hfield_data[0] = np.inf
    self.assertEqual(self.lib.mjf_enableHfieldMip(m._address, 0, 1), -1)
    self.assertEqual(m.hfield_mipvalid[0], 0)

  def test_copy_and_binary_roundtrip_own_their_pyramids(self):
    m = make_model()
    m.hfield_data[:] = np.linspace(0, 1, m.nhfielddata)
    self.lib.mjf_enableHfieldMip(m._address, 0, 1)
    copied = copy.copy(m)
    with tempfile.TemporaryDirectory() as tmp:
      path = str(Path(tmp)/'model.mjb')
      mujoco.mj_saveModel(m, path)
      loaded = mujoco.MjModel.from_binary_path(path)
    for other in (copied, loaded):
      self.assertEqual(other.hfield_mipvalid[0], 1)
      np.testing.assert_array_equal(other.hfield_mipmin, m.hfield_mipmin)
      np.testing.assert_array_equal(other.hfield_mipmax, m.hfield_mipmax)
      self.assertFalse(np.shares_memory(other.hfield_mipmax, m.hfield_mipmax))

  def test_live_write_after_rejection_and_trajectory_parity(self):
    for kind in ('box', 'mesh'):
      m = make_model(kind=kind); d = mujoco.MjData(m)
      ref = copy.copy(m); r = mujoco.MjData(ref)
      self.lib.mjf_enableHfieldMip(m._address, 0, 1)
      d.qpos[2] = r.qpos[2] = .5
      mujoco.mj_forward(m, d)
      self.assertEqual(d.ncon, 0)
      for step in range(200):
        if step in (0, 30, 50):
          row = 30
          data = np.full(5*67, .4 + step/100, np.float32)
          self.lib.mjf_writeHfieldRows(m._address, d._address, 0, row, 5,
                                      data.ctypes.data, data.size)
          ref.hfield_data.reshape(65, 67)[row:row+5] = data.reshape(5, 67)
        mujoco.mj_step(m, d); mujoco.mj_step(ref, r)
        for got, expected in zip(state(d), state(r)):
          np.testing.assert_array_equal(got, expected)
      self.assertGreater(d.ncon, 0)


if __name__ == '__main__':
  unittest.main()

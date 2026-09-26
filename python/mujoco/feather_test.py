# Copyright 2026 Feather contributors
# Licensed under the Apache License, Version 2.0.
"""Native Feather timer contract; run against a complete matching fork wheel."""
from concurrent.futures import ThreadPoolExecutor
import ctypes
from pathlib import Path
import sys
import time
import unittest

import mujoco
import numpy as np


XML = '''<mujoco><option timestep=".002" integrator="implicitfast"/>
  <worldbody><geom type="plane" size="5 5 .1"/>
    <body pos="0 0 .099"><freejoint/><geom type="box" size=".1 .1 .1" mass="1"/>
      <body pos="0 0 .3"><joint name="hinge" axis="0 1 0"/>
        <geom type="capsule" size=".04 .12" mass=".1"/></body>
    </body></worldbody><actuator><motor joint="hinge"/></actuator></mujoco>'''


class FeatherTimerTest(unittest.TestCase):

  def setUp(self):
    version = mujoco.mj_versionString()
    filename = ('mujoco.dll' if sys.platform == 'win32' else
                f'libmujoco.{version}.dylib' if sys.platform == 'darwin' else
                f'libmujoco.so.{version}')
    self.lib = ctypes.CDLL(str(Path(mujoco.__file__).parent / filename))
    self.enable = self.lib.mjf_enableTimers
    self.enable.argtypes = [ctypes.c_int]
    self.enable.restype = None
    self.slot = ctypes.c_void_p.in_dll(self.lib, 'mjcb_time')
    self.old_pointer = self.slot.value
    self.old_python = mujoco.get_mjcb_time()
    mujoco.set_mjcb_time(None)
    self.m = mujoco.MjModel.from_xml_string(XML)

  def tearDown(self):
    # Restore both the Python owner reference and the actual C callback slot.
    mujoco.set_mjcb_time(self.old_python)
    self.slot.value = self.old_pointer

  def test_durations_counts_and_disable_preserve_upstream_semantics(self):
    d = mujoco.MjData(self.m)
    self.enable(1)
    mujoco.mj_step(self.m, d, nstep=200)
    timer = mujoco.mjtTimer
    self.assertEqual(d.timer.number[timer.mjTIMER_STEP], 200)
    self.assertEqual(d.timer.number[timer.mjTIMER_INVERSE], 0)
    active = d.timer.number > 0
    self.assertEqual(np.count_nonzero(active), int(timer.mjNTIMER)-1)
    self.assertTrue((d.timer.duration[active] > 0).all())
    self.assertTrue((d.timer.duration[active] <= d.timer.duration[timer.mjTIMER_STEP]).all())
    before, counts = d.timer.duration.copy(), d.timer.number.copy()
    self.enable(0)
    self.assertIsNone(self.slot.value)
    np.testing.assert_array_equal(d.timer.duration, before)
    np.testing.assert_array_equal(d.timer.number, counts)
    mujoco.mj_step(self.m, d, nstep=10)
    np.testing.assert_array_equal(d.timer.duration, before)
    self.assertEqual(d.timer.number[timer.mjTIMER_STEP], 210)
    d.timer.duration[:] = 0
    d.timer.number[:] = 0
    self.enable(1)
    mujoco.mj_step(self.m, d)
    self.assertEqual(d.timer.number[timer.mjTIMER_STEP], 1)
    self.assertGreater(d.timer.duration[timer.mjTIMER_STEP], 0)

  def test_clock_is_monotonic_wall_time_in_milliseconds(self):
    self.enable(1)
    clock = ctypes.CFUNCTYPE(ctypes.c_double)(self.slot.value)
    values = [clock() for _ in range(1000)]
    self.assertTrue(np.isfinite(values).all())
    self.assertTrue((np.diff(values) >= 0).all())
    outer_start = time.monotonic()
    native_start = clock()
    inner_start = time.monotonic()
    time.sleep(.02)
    inner_end = time.monotonic()
    native_end = clock()
    outer_end = time.monotonic()
    elapsed = native_end-native_start
    self.assertGreaterEqual(elapsed, 1000*(inner_end-inner_start)-.01)
    self.assertLessEqual(elapsed, 1000*(outer_end-outer_start)+.01)

  def test_replaces_python_callback_and_disable_does_not_restore_it(self):
    calls = []
    def callback():
      calls.append(1)
      return time.monotonic()*1000
    d = mujoco.MjData(self.m)
    mujoco.set_mjcb_time(callback)
    mujoco.mj_step(self.m, d)
    self.assertGreater(len(calls), 0)
    before = len(calls)
    self.enable(1)
    mujoco.mj_step(self.m, d)
    self.enable(0)
    mujoco.mj_step(self.m, d)
    self.assertEqual(len(calls), before)
    self.assertIsNone(self.slot.value)
    # Reinstall explicitly; the native API does not own Python callback lifetime.
    mujoco.set_mjcb_time(callback)
    mujoco.mj_step(self.m, d)
    self.assertGreater(len(calls), before)

  def test_shared_clock_disjoint_worker_data_matches_untimed_state(self):
    datas = [mujoco.MjData(self.m) for _ in range(4)]
    refs = [mujoco.MjData(self.m) for _ in datas]
    for i, (d, ref) in enumerate(zip(datas, refs)):
      d.qvel[0] = ref.qvel[0] = .1*i
      d.ctrl[0] = ref.ctrl[0] = .03*i
      mujoco.mj_step(self.m, ref, nstep=200)
    self.enable(1)
    def step(d):
      mujoco.mj_step(self.m, d, nstep=200)
    with ThreadPoolExecutor(max_workers=4) as pool:
      list(pool.map(step, datas))
    self.enable(0)  # all workers have joined
    for d, ref in zip(datas, refs):
      for field in ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'efc_force'):
        np.testing.assert_array_equal(getattr(d, field), getattr(ref, field))
      np.testing.assert_array_equal(d.contact.frame, ref.contact.frame)
      np.testing.assert_array_equal(d.contact.dist, ref.contact.dist)
      self.assertEqual(d.timer.number[mujoco.mjtTimer.mjTIMER_STEP], 200)
      self.assertGreater(d.timer.duration[mujoco.mjtTimer.mjTIMER_STEP], 0)


if __name__ == '__main__':
  unittest.main()

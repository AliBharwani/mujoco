# Copyright 2026 Feather contributors
# Licensed under the Apache License, Version 2.0.
"""Regression tests for the Feather polyhedral heightfield contact path.

Run against Python bindings linked to this fork, e.g. with PYTHONPATH pointing
at a staged 3.9 package containing the rebuilt libmujoco.so.3.9.0.
"""
import itertools
import unittest

import mujoco
import numpy as np


VERTICES = ' '.join(str(v) for xyz in itertools.product(
    (-.125, .125), (-.05, .05), (-.02, .02)) for v in xyz)


def model(kind='mesh', n=17, native=True, surface='hfield', hquat='1 0 0 0'):
  geom = 'type="box" size=".125 .05 .02"' if kind == 'box' else 'type="mesh" mesh="sole"'
  floor = 'type="hfield" hfield="h"' if surface == 'hfield' else 'type="plane" size="2 2 .1"'
  return mujoco.MjModel.from_xml_string(f'''<mujoco>
    <option timestep=".0022222222222222222" integrator="implicitfast">
      <flag nativeccd="{'enable' if native else 'disable'}"/></option>
    <default><geom margin=".001" solref=".015 1" solimp=".99 .99 .003"
      friction="1 .005 .0001"/></default>
    <asset><mesh name="sole" vertex="{VERTICES}"/>
      <hfield name="h" nrow="{n}" ncol="{n}" size="1.25 1.25 1 .1"/></asset>
    <worldbody><geom name="floor" {floor} quat="{hquat}"/>
      <body pos=".013 .037 .02"><freejoint/><geom {geom} mass="10"/></body>
    </worldbody></mujoco>''')


class HfieldTopSurfaceTest(unittest.TestCase):

  def test_flat_grid_seams_and_ccd_backends(self):
    for kind, native, n in itertools.product(('box', 'mesh'), (True, False), (9, 17, 33)):
      m = model(kind, n=n, native=native)
      d = mujoco.MjData(m)
      for yaw, x, y in itertools.product((0, .31, .78), np.linspace(0, .15625, 7),
                                         np.linspace(0, .15625, 7)):
        d.qpos[:] = [x, y, .019, np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
        mujoco.mj_forward(m, d)
        self.assertEqual(d.ncon, 4, (kind, native, n, x, y, yaw))
        for c in d.contact:
          np.testing.assert_allclose(c.frame[:3], [0, 0, 1], atol=1e-9)
          self.assertAlmostEqual(c.pos[2] - .5*c.dist, 0, places=10)
          self.assertAlmostEqual(c.dist, -.001, places=7)

  def test_five_second_passive_drift(self):
    for kind, surface, yaw in itertools.product(('box', 'mesh'), ('plane', 'hfield'), (0, .31, .78)):
      m = model(kind, surface=surface)
      d = mujoco.MjData(m)
      d.qpos[3:] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
      # Exclude placement/settling transient, measure five seconds at rest.
      for _ in range(450):
        mujoco.mj_step(m, d)
      start = d.qpos[:2].copy()
      for _ in range(2250):
        mujoco.mj_step(m, d)
      # Plane-vs-mesh uses a different, three-point manifold and can itself
      # drift slightly. The requested 1 mm limit applies to the fixed hfield.
      if surface == 'hfield':
        self.assertLess(np.linalg.norm(d.qpos[:2] - start), .001, (kind, surface, yaw))
      self.assertTrue(np.isfinite(d.qpos).all())
      self.assertFalse(d.warning.number.any())

  def test_slopes_have_terrain_normal_and_consistent_witness(self):
    for kind, slope in itertools.product(('box', 'mesh'), (.1, .7, 3.0)):
      m = model(kind)
      xy = np.linspace(-1.25, 1.25, 17)
      # Bound the ramp to the declared elevation, leaving central cells planar.
      m.hfield_data[:] = np.clip(.5 + slope*xy[None, :] + np.zeros((17, 1)), 0, 1).flat
      d = mujoco.MjData(m)
      d.qpos[:3] = [0, 0, .5 + .019]
      mujoco.mj_forward(m, d)
      normal = np.array([-slope, 0, 1]) / np.sqrt(1 + slope*slope)
      self.assertGreater(d.ncon, 0)
      for c in d.contact:
        np.testing.assert_allclose(c.frame[:3], normal, atol=2e-6)
        terrain = c.pos - .5*c.dist*c.frame[:3]
        self.assertAlmostEqual(np.dot(terrain, normal), .5*normal[2], places=6)

  def test_rotated_heightfield(self):
    theta = .23
    q = np.array([np.cos(theta/2), np.sin(theta/2), 0, 0])
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, q)
    rotation = rotation.reshape(3, 3)
    for kind in ('box', 'mesh'):
      m = model(kind, hquat=' '.join(map(str, q)))
      d = mujoco.MjData(m)
      d.qpos[:3] = rotation @ np.array([.013, .037, .019])
      d.qpos[3:] = q
      mujoco.mj_forward(m, d)
      self.assertEqual(d.ncon, 4)
      for c in d.contact:
        np.testing.assert_allclose(c.frame[:3], rotation[:, 2], atol=1e-9)
        self.assertAlmostEqual(np.dot(c.pos, rotation[:, 2]) - .5*c.dist, 0, places=9)

  def test_positive_margin_and_separation(self):
    for kind in ('box', 'mesh'):
      m = model(kind)
      d = mujoco.MjData(m)
      for bottom, count in ((.0015, 4), (.003, 0)):
        d.qpos[2] = .02 + bottom
        mujoco.mj_forward(m, d)
        self.assertEqual(d.ncon, count)
        for c in d.contact:
          self.assertAlmostEqual(c.dist, bottom, places=7)

  def test_above_footprint_uses_normal_distance_margin_on_steep_ridge(self):
    # Bottom is above the highest sample by .025, exceeding the .02 pair
    # margin vertically, but within that margin along the slope's normal.
    # A max-height + margin rejection would incorrectly discard these contacts.
    vertices = ' '.join(str(v) for xyz in itertools.product(
        (-.008, .008), (-.03, .03), (-.002, .002)) for v in xyz)
    for kind in ('box', 'mesh'):
      geom = ('type="box" size=".008 .03 .002"' if kind == 'box'
              else 'type="mesh" mesh="sole"')
      m = mujoco.MjModel.from_xml_string(f'''<mujoco>
        <default><geom margin=".01"/></default>
        <asset><hfield name="h" nrow="3" ncol="5" size="1 .5 1 .1"/>
          <mesh name="sole" vertex="{vertices}"/></asset>
        <worldbody><geom type="hfield" hfield="h"/>
          <body pos="0 0 1.027"><freejoint/><geom {geom} mass="1"/></body>
        </worldbody></mujoco>''')
      m.hfield_data[:] = np.tile([0, 0, 1, 0, 0], 3)
      d = mujoco.MjData(m)
      mujoco.mj_forward(m, d)
      self.assertGreater(d.ncon, 0, kind)
      self.assertTrue((d.contact.dist > 0).all())
      self.assertTrue((d.contact.dist <= .02).all())
      np.testing.assert_allclose(d.contact.frame[:, 2], 1/np.sqrt(5), atol=1e-12)

  def test_above_footprint_rechecks_live_height_writes(self):
    for kind, n in itertools.product(('box', 'mesh'), (17, 129)):
      m = model(kind, n=n)
      d = mujoco.MjData(m)
      d.qpos[2] = .08  # lowest point .06, within the declared hfield volume
      for height, expected in ((0, 0), (.061, 4), (0, 0)):
        m.hfield_data[:] = height
        mujoco.mj_forward(m, d)
        self.assertEqual(d.ncon, expected, (kind, n, height))
        if expected:
          np.testing.assert_allclose(d.contact.dist, -.001, atol=2e-8)
          np.testing.assert_allclose(d.contact.frame[:, 2], 1, atol=1e-12)

  def test_ridge_and_valley_normals_belong_to_top_faces(self):
    for kind, sign in itertools.product(('box', 'mesh'), (-1, 1)):
      m = model(kind)
      xy = np.linspace(-1.25, 1.25, 17)
      m.hfield_data[:] = (.5 + sign*.2*np.abs(xy)[None, :] + np.zeros((17, 1))).flat
      d = mujoco.MjData(m)
      d.qpos[:3] = [0, 0, .519]
      mujoco.mj_forward(m, d)
      expected = [np.array([s*.2, 0, 1])/np.sqrt(1.04) for s in (-1, 1)]
      self.assertGreater(d.ncon, 0)
      for c in d.contact:
        self.assertLess(min(np.linalg.norm(c.frame[:3] - v) for v in expected), 2e-6)
        self.assertLessEqual(c.dist, .002)

  def test_outer_walls_and_underside_retain_solid_contacts(self):
    for kind in ('box', 'mesh'):
      m = model(kind)
      d = mujoco.MjData(m)
      for pos in ((1.35, .07, -.04), (.07, .07, -.115)):
        d.qpos[:3] = pos
        mujoco.mj_forward(m, d)
        self.assertGreater(d.ncon, 0)
        self.assertTrue(any(c.frame[2] < .9 for c in d.contact))
        self.assertTrue(np.isfinite(d.contact.pos).all())

  def test_arbitrary_orientations_and_dense_nonplanar_fields(self):
    rng = np.random.default_rng(812)
    for kind, flat in itertools.product(('box', 'mesh'), (True, False)):
      m = model(kind, n=65)
      m.hfield_data[:] = .5 if flat else rng.uniform(.45, .55, m.hfield_data.size)
      d = mujoco.MjData(m)
      for _ in range(150):
        q = rng.normal(size=4)
        d.qpos[3:] = q / np.linalg.norm(q)
        d.qpos[:3] = [*rng.uniform(-.5, .5, 2), .5]
        mujoco.mj_forward(m, d)
        self.assertLessEqual(d.ncon, 50)
        self.assertGreater(d.ncon, 0)
        self.assertTrue(np.isfinite(d.contact.pos).all())
        self.assertTrue(np.isfinite(d.contact.dist).all())
        if flat:
          for c in d.contact:
            np.testing.assert_allclose(c.frame[:3], [0, 0, 1], atol=1e-9)
        else:
          self.assertTrue((d.contact.frame[:, 2] > 0).all())


if __name__ == '__main__':
  unittest.main()

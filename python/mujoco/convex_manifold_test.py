# Copyright 2026 Feather contributors. Licensed under Apache-2.0.
"""Analytic positive-margin face clipping and fallbacks on actual native calls."""
import itertools
import unittest
import mujoco as mj
import numpy as np


def vertices(size):
  return ' '.join(str(v) for xyz in itertools.product(*[(-s, s) for s in size]) for v in xyz)


def model(floor='box', hull='mesh', margin=.01, flags='', size=(.2,.3,.1)):
  def geom(kind, name, sz):
    return f'type="mesh" mesh="{name}"' if kind == 'mesh' else f'type="box" size="{sz}"'
  return mj.MjModel.from_xml_string(f'''<mujoco><compiler angle="radian"/>
    <option gravity="0 0 0"><flag {flags}/></option>
    <default><geom margin="{margin}"/></default>
    <asset><mesh name="floor" vertex="{vertices((1,1,.1))}"/>
      <mesh name="hull" vertex="{vertices(size)}"/></asset>
    <worldbody><geom name="floor" {geom(floor,'floor','1 1 .1')} pos="0 0 -.1"/>
      <body name="hull" pos="0 0 {size[2]}"><freejoint/>
        <geom name="hull" {geom(hull,'hull',' '.join(map(str,size)))} mass="1"/>
      </body></worldbody></mujoco>''')


class ConvexManifoldTest(unittest.TestCase):
  def test_parallel_faces_at_separation_and_penetration(self):
    for floor,hull,margin,gap in itertools.product(('mesh','box'),('mesh','box'),(.001,.02),(-.0005,.0005)):
      m=model(floor,hull,margin);d=mj.MjData(m);d.qpos[2]+=gap
      mj.mj_forward(m,d)
      self.assertEqual(d.ncon,4,(floor,hull,margin,gap,d.ncon))
      np.testing.assert_allclose(d.contact.dist,gap,atol=3e-8,rtol=0)
      np.testing.assert_allclose(d.contact.pos[:,2],gap/2,atol=3e-8,rtol=0)
      expected=np.array([[0,0,1 if c.geom[0] == 0 else -1] for c in d.contact])
      np.testing.assert_allclose(d.contact.frame[:,:3],expected,atol=1e-8,rtol=0)
      np.testing.assert_allclose(np.sort(np.abs(d.contact.pos[:,:2]),axis=0),np.tile([.2,.3],(4,1)),atol=3e-8,rtol=0)

  def test_near_margin_coincident_witnesses_keep_the_face_normal(self):
    # Binary-exact mesh extents put two tilted vertices within a few ulps of
    # the margin. Their inflated witnesses are too close to normalize reliably.
    m=model(margin=.015625,size=(.25,.25,.125));d=mj.MjData(m)
    for angle in (.0001,.0005,.001):
      d.qpos[3:]=[np.cos(angle/2),0,np.sin(angle/2),0]
      d.qpos[2]=.03125+.125*np.cos(angle)-.25*np.sin(angle)
      mj.mj_forward(m,d);self.assertEqual(d.ncon,4)
      self.assertTrue(np.any(np.abs(d.contact.dist-.03125)<1e-14))
      np.testing.assert_allclose(d.contact.frame[:,:3],np.tile([0,0,1],(4,1)),atol=1e-8,rtol=0)

  def test_outside_margin_does_not_invent_contacts(self):
    for floor in ('box','mesh'):
      m=model(floor);d=mj.MjData(m);d.qpos[2]+=.021
      mj.mj_forward(m,d);self.assertEqual(d.ncon,0)

  def test_translation_yaw_and_contact_normal_order(self):
    for floor,hull in (('box','mesh'),('mesh','box'),('mesh','mesh')):
      m=model(floor,hull);d=mj.MjData(m)
      for yaw in (0,.1,.4,1.1):
        d.qpos[:3]=[.15,-.12,.1005]
        d.qpos[3:]=[np.cos(yaw/2),0,0,np.sin(yaw/2)]
        mj.mj_forward(m,d);self.assertEqual(d.ncon,4)
        np.testing.assert_allclose(d.contact.dist,.0005,atol=3e-8,rtol=0)
        # Signed normal always points from geom[0] toward geom[1].
        for c in d.contact:
          self.assertGreater(np.dot(c.frame[:3],d.geom_xpos[c.geom[1]]-d.geom_xpos[c.geom[0]]),0)

  def test_tilted_face_witnesses_have_geometric_depth(self):
    m=model();d=mj.MjData(m);angle=.0005
    d.qpos[:3]=[0,0,.1005]
    d.qpos[3:]=[np.cos(angle/2),0,np.sin(angle/2),0]
    mj.mj_forward(m,d);self.assertEqual(d.ncon,4)
    # Resolve both witness points from the midpoint and signed distance.
    for c in d.contact:
      a=c.pos-.5*c.dist*c.frame[:3];b=c.pos+.5*c.dist*c.frame[:3]
      if c.geom[0] != 0:a,b=b,a
      self.assertAlmostEqual(a[2],0,places=8)
      # Mesh compiler changes its local frame; compare to authored body frame.
      local=d.xmat[1].reshape(3,3).T@(b-d.xpos[1])
      self.assertAlmostEqual(local[2],-.1,places=7)
      self.assertLess(c.dist,.01)

  def test_unaligned_edge_uses_finite_single_contact(self):
    m=model(size=(.2,.3,.1));d=mj.MjData(m);angle=.4
    d.qpos[:3]=[0,0,.1*np.cos(angle)+.3*np.sin(angle)-.0005]
    d.qpos[3:]=[np.cos(angle/2),np.sin(angle/2),0,0]
    mj.mj_forward(m,d);self.assertEqual(d.ncon,1)
    self.assertTrue(np.all(np.isfinite(d.contact.frame)))
    self.assertLess(d.contact.dist[0],0)

  def test_multiccd_disabled_stays_single(self):
    m=model(flags='multiccd="disable"');d=mj.MjData(m);d.qpos[2]+=.0005
    mj.mj_forward(m,d);self.assertEqual(d.ncon,1)

  def test_libccd_fallback_stays_available(self):
    m=model(flags='nativeccd="disable"');d=mj.MjData(m);d.qpos[2]-=.0005
    mj.mj_forward(m,d);self.assertGreater(d.ncon,0);self.assertLessEqual(d.ncon,5)
    self.assertTrue(np.all(np.isfinite(d.contact.pos)))

  def test_contact_reduction_respects_cap(self):
    m=model('mesh','mesh',size=(.8,.8,.1));d=mj.MjData(m)
    d.qpos[3:]=[np.cos(np.pi/8),0,0,np.sin(np.pi/8)]
    d.qpos[2]+=.0005
    mj.mj_forward(m,d);self.assertEqual(d.ncon,4)
    np.testing.assert_allclose(d.contact.dist,.0005,atol=3e-8,rtol=0)
    self.assertEqual(len(np.unique(np.round(d.contact.pos,8),axis=0)),4)

  def test_zero_margin_keeps_native_face_clipping(self):
    m=model(margin=0);d=mj.MjData(m);d.qpos[2]-=.0005
    mj.mj_forward(m,d);self.assertEqual(d.ncon,4)
    np.testing.assert_allclose(d.contact.dist,-.0005,atol=3e-8,rtol=0)


if __name__ == '__main__':unittest.main()

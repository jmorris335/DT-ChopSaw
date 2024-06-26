"""
| File: skeleton.py 
| Info: Methods for constructing the configuration of the saw based on a set of markers
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, 7 Jun 2024: Initialized
"""
import yaml
from types import SimpleNamespace
import numpy as np

if __name__ == '__main__':
    import os, sys
    os.chdir("/Users/john_morris/Documents/Clemson/Research/DTs-Public/DT-ChopSaw")
    sys.path.insert(0, '')

from src.gui.plotter3D import makeCylinderFromEndpoints, makeCylinder
from src.auxiliary.transform import Transform
from src.auxiliary.geometry import pointDistance
 
def plotSaw(origin=(0.,0.,0.), miter_angle=0., bump_offset=(0.,0.,0.), 
            bevel_angle=0., slider_offset=0., crash_angle=0., params=None, **kwargs):
    """Plots the saw with the specified state variables."""
    if params is None:
        params = gatherSawParameters()
    params.update({"origin": origin, "miter_angle": miter_angle, 
                   "bump_offset": bump_offset, 'bevel_angle': bevel_angle, 
                   'slider_offset': slider_offset, 'crash_angle': crash_angle})
    params.update(kwargs)
    params = SimpleNamespace(**params)
    return getSawSurfaces(params)

def gatherSawParameters(sawinfo_filepath: str="saw_info.yaml") -> dict:
    """Parses the parameters file and forms a dictionary with all members."""
    with open(sawinfo_filepath, 'r') as file:
        saw_info, blade_info = yaml.safe_load_all(file)
    saw_dims = SimpleNamespace(**saw_info['dims'])
    blade_dims = SimpleNamespace(**blade_info['dims'])
    params = {
        'base_radius': saw_dims.radius_base,
        'base_thickness': saw_dims.thickness_base,
        'base_color': '#303030',
        'stem_radius': saw_dims.radius_stem,
        'stem_height': saw_dims.height_stem,
        'stem_offset': saw_dims.offset_stem,
        'stem_color': '#303030',
        'slider_radius': saw_dims.radius_slider,
        'slider_length': saw_dims.length_slider,
        'slider_color': '#BBBBBB',
        'arm_radius': saw_dims.radius_arm,
        'arm_length': saw_dims.length_arm,
        'arm_color': '#F3B627',
        'blade_radius': blade_dims.radius_blade,
        'blade_thickness': blade_dims.thickness_blade,
        'blade_color': '#993333'
    }
    return params

def getSawSurfaces(p: SimpleNamespace) -> list:
    """Finds and returns the surfaces for each member of the saw."""
    pts = getSawChain(p.origin, p.bump_offset, p.miter_angle, p.bevel_angle, 
                         p.slider_offset, p.crash_angle, p.stem_offset, p.stem_height, 
                         p.slider_length, p.arm_length, p.blade_thickness)
    miter_COR, bevel_COR, stem_top, arm_COR, blade_COR, blade_end = pts
    print("\n" + str(pts) + "\n")
    base = makeBase(miter_COR, p.base_radius, p.base_thickness, p.base_color)
    stem = makeStem(bevel_COR, stem_top, p.stem_radius, p.stem_color)
    slider = makeSlider(arm_COR, stem_top, p.slider_radius, p.slider_length, p.slider_color)
    arm = makeArm(arm_COR, blade_COR, p.arm_radius, p.arm_color)
    blade = makeBlade(blade_COR, blade_end, p.blade_radius, p.blade_thickness, p.blade_color)
    return base + stem + slider + arm + blade

def makeBase(center, radius, height, color):
    """Returns the surfaces for the base of the saw"""
    center[2] = center[2] - height #artificially lower base
    return makeCylinder(*center, radius, height, n_elements=12, color=color)

def makeStem(bevel_COR, stem_top, radius, color):
    """Returns the surfaces for the stem of the saw"""
    return makeCylinderFromEndpoints(bevel_COR, stem_top, radius, n_elements=5, color=color)

def makeSlider(arm_COR, stem_top, radius, length, color):
    """Returns the surfaces for the slider of the saw"""
    t = length / pointDistance(arm_COR, stem_top)
    end_pt = [arm_COR[i] + t * (stem_top[i] - arm_COR[i]) for i in range(len(arm_COR))]
    return makeCylinderFromEndpoints(end_pt, arm_COR, radius, n_elements=12, color=color)

def makeArm(arm_COR, blade_COR, radius, color):
    """Returns the surfaces for the arm of the saw"""
    return makeCylinderFromEndpoints(arm_COR, blade_COR, radius, n_elements=5, color=color)

def makeBlade(blade_COR, blade_end, radius, thickness, color):
    """Returns the surfaces for the blade of the saw"""
    t = thickness / pointDistance(blade_COR, blade_end)
    blade_start = [blade_COR[i] + t * (blade_end[i] - blade_COR[i]) for i in range(len(blade_COR))]
    return makeCylinderFromEndpoints(blade_start, blade_end, radius, n_elements=18, color=color)

def getSawChain(origin, bump_offset, miter_angle, bevel_angle, slider_offset, 
                   crash_angle, stem_offset, stem_height, slider_length, 
                   arm_length, blade_thickness):
    """Returns an array of points for each member intersection in the following 
    order: [miter_COR, bevel_COR, stem_top, arm_COR, blade_COR, blade_start]. 
    """
    origin = np.array(origin)
    T = Transform(centroid=origin)
    T.translate(*bump_offset)
    miter_COR = T.transform(origin)[0][:3]

    T = Transform(centroid=origin)
    T.translate(*bump_offset)
    T.translate(dely=-stem_offset)
    T.rotate(psi=miter_angle, x=miter_COR[0], y=miter_COR[1], z=miter_COR[2])
    bevel_COR = T.transform(origin)[0][:3]

    T = Transform(centroid=origin)
    T.translate(*bump_offset)
    T.translate(dely=-stem_offset)
    T.translate(delz=stem_height)
    T.rotate(phi=bevel_angle, x=bevel_COR[0], y=bevel_COR[1], z=bevel_COR[2])
    T.rotate(psi=miter_angle, x=miter_COR[0], y=miter_COR[1], z=miter_COR[2])
    stem_top = T.transform(origin)[0][:3]

    T = Transform(centroid=origin)
    T.translate(*bump_offset)
    T.translate(dely=-stem_offset)
    T.translate(delz = stem_height)
    T.translate(dely = slider_length + slider_offset)
    T.rotate(psi=miter_angle, x=miter_COR[0], y=miter_COR[1], z=miter_COR[2])
    T.rotate(phi=bevel_angle, x=bevel_COR[0], y=bevel_COR[1], z=bevel_COR[2])
    arm_COR = T.transform(origin)[0][:3]

    T = Transform(centroid=origin)
    T.translate(*bump_offset)
    T.translate(dely=-stem_offset)
    T.translate(delz = stem_height)
    T.translate(dely = slider_length + slider_offset)
    T.translate(dely = arm_length)
    T.rotate(theta=-crash_angle, x=arm_COR[0], y=arm_COR[1], z=arm_COR[2])
    T.rotate(phi=bevel_angle, x=bevel_COR[0], y=bevel_COR[1], z=bevel_COR[2])
    T.rotate(psi=miter_angle, x=miter_COR[0], y=miter_COR[1], z=miter_COR[2])
    blade_COR = T.transform(origin)[0][:3]

    T = Transform(centroid=origin)
    T.translate(*bump_offset)
    T.translate(dely=-stem_offset)
    T.translate(delz = stem_height)
    T.translate(dely = slider_length + slider_offset)
    T.translate(dely = arm_length)
    T.translate(delx = -blade_thickness/2)
    T.rotate(theta=-crash_angle, x=arm_COR[0], y=arm_COR[1], z=arm_COR[2])
    T.rotate(phi=bevel_angle, x=bevel_COR[0], y=bevel_COR[1], z=bevel_COR[2])
    T.rotate(psi=miter_angle, x=miter_COR[0], y=miter_COR[1], z=miter_COR[2])
    blade_end = T.transform(origin)[0][:3]
    return [miter_COR, bevel_COR, stem_top, arm_COR, blade_COR, blade_end]

def getSawChainOld(Ts=None, thetas=None, stem_offset=None, stem_height=None, 
                slider_length=None, arm_length=None):
    """Returns an array of points for each member intersection in the following 
    order: [base_COR, stem_bottom, stem_top, arm_COR, blade_COR]. 

    Parameters
    ----------
    Ts : list, Optional
        List of 4x4 transformation matrices from spatial to ith frame. If not 
        provided, then automatically calculated via the thetas.
    thetas : list, Optional
        A list of joint displacements (angles, translations) for each joint. Only
        used to calculate `Ts` if `Ts` is not provided.
    stem_offset, stem_height, slider_lenggth, arm_length : float, Optional
        The lengths of each chain. Only necessary if `Ts` is not provided.

    Note that if `Ts` is given, `thetas` and `kwargs` are not necessary.
    """
    if Ts is None:
        Ts = getSawTransformations(thetas, stem_offset, stem_height, 
                                   slider_length, arm_length)
    pts = [T[1:3, :] for T in Ts]
    return pts
        
def getSawTransformations(thetas, stem_offset, stem_height, slider_length, arm_length):
    """Returns the spatial transformation matrices for transforming the spatial 
    frame at to the frame collocated with joint `i` for a given theta."""
    Ms = getInitialSawState(stem_offset, stem_height, slider_length, arm_length)
    screws = getSawScrewAxes(stem_offset, stem_height, slider_length, arm_length)

    expsProduct = np.eye(4)
    Ts = list()
    for i in range(len(thetas)):
        exp_S_Theta = matrixExponential(screws[i], thetas[i])
        expsProduct = expsProduct * exp_S_Theta
        Ts.append(expsProduct * Ms[i])
    return Ts

def getInitialSawState(stem_offset, stem_height, slider_length, arm_length):
    """Returns a list of 4x4 matrices detailing the initial rotations and 
    displacements of each frame.
    """
    return [
        np.eye(4),
        makeInitialState((-stem_offset, 0, 0)),
        makeInitialState((-stem_offset, stem_height, 0)),
        makeInitialState((slider_length - stem_offset, stem_height, 0)),
        makeInitialState((arm_length + slider_length - stem_offset, stem_height, 0))
    ]

def makeInitialState(position):
    """Returns a 4x4 transformation matrix representing an unrotated frame 
    collocated at the point `position`.
    """
    RandP = np.hstack((np.eye(3), np.array([position]).T))
    augment = np.vstack((RandP, np.array([0, 0, 0, 1])))
    return augment

def getSawScrewAxes(stem_offset, stem_height, slider_length, arm_length):
    return np.array([
        [0, 1, 0, 0, 0, 0], #S1, miter
        [1, 0, 0, 0, 0, 0], #S2, bevel
        [0, 0, 0, 1, 0, 0], #S3, slider
        [0, 0, 1, stem_height, slider_length - stem_offset, 0], #S4, crash
        [0, 0, -1, stem_height, stem_offset - slider_length - arm_length, 0] #S5, blade rotation
    ])

def matrixExponential(S, theta):
    """Calculates the matrix exponential for a 1x6 array `S` multiplied by a 
    constant `theta`"""
    if not isinstance(S, np.ndarray): 
        S = np.array(S)
    omega = S[1:3]
    v = S[4:6].T
    o_hat = hat(omega)
    R = np.eye(3) + np.sin(theta) * o_hat + (1 - np.cos(theta)) * o_hat^2
    G = np.eye(3)*theta + (1 - np.cos(theta)) * o_hat + (theta - np.sin(theta)) * o_hat^2
    temp = G @ v
    return np.array([[R, temp], [0, 0, 0, 1]])

def hat(a):
    """Returns the 3x3 skew-symmetrix matrix representing the vector `a`"""
    return np.array([[0,      -a[3],    a[2]],
                     [a[3],    0,      -a[1]],
                     [-a[2],   a[1],    0]])


def skeletonTest():
    info = gatherSawParameters()
    import plotly.graph_objects as go
    surfaces = plotSaw(origin=(0,0,0), miter_angle=-np.pi/4, bump_offset=(0., 0., 0.), 
                       bevel_angle=np.pi/6, slider_offset=-0.1, crash_angle=np.pi/4)
    fig = go.Figure(surfaces)
    fig.update_layout(
    scene = dict(
        aspectmode='data',
        aspectratio=dict(x=1, y=1, z=1)),
    margin=dict(r=20, l=10, b=10, t=10))
    fig.show()

if __name__ == '__main__':
    skeletonTest()
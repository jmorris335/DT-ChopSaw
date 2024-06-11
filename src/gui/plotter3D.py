"""
| File: plotter3D.py 
| Info: Methods for 3D plotting using the Plotly framework
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization

| Version History:
| - 0.0, 5 Jun 2024: Initialized
"""
import numpy as np
import plotly.graph_objects as go

from src.auxiliary.geometry import pointDistance

if __name__ == '__main__':
    import os, sys
    os.chdir("/Users/john_morris/Documents/Clemson/Research/DTs-Public/DT-ChopSaw")
    sys.path.insert(0, '')

from src.auxiliary.transform import Transform

def makeCylinderFromEndpoints(pt1, pt2, radius, color, opacity=1.0, n_elements=15):
    """Wrapper for makeCylinder that takes a set of endpoints and returns the 
    surfaces of the cylinder."""
    dz = pointDistance(pt1, pt2)
    adj_pt = [pt2[i] - pt1[i] for i in range(len(pt1))]
    adj_pt = adj_pt / np.linalg.norm(adj_pt)
    x, y, z = adj_pt
    psi = np.arctan2(y, x)
    phi = np.arctan2(np.sqrt(x**2 + y**2), z)
    return makeCylinder(*pt1, radius, dz, 0., phi, psi, n_elements, color, opacity)

def makeCylinderWall(x, y, z, r, dz, n_elements=15):
    """Create a cylindrical mesh located at x, y, z, with radius r and height dz
    Taken from https://stackoverflow.com/a/72183849
    """
    center_z = np.linspace(0, dz, n_elements)
    theta = np.linspace(0, 2*np.pi, n_elements)
    theta_grid, z_grid = np.meshgrid(theta, center_z)
    x_grid = r * np.cos(theta_grid) + x
    y_grid = r * np.sin(theta_grid) + y
    z_grid = z_grid + z
    return x_grid, y_grid, z_grid

def makeCircle(x, y, z, r, n_elements):
    """Create a circular mesh located at x, y, z with radius r
    Taken from https://stackoverflow.com/a/72183849
    """
    r_discr = np.linspace(0, r, 2)
    theta_discr = np.linspace(0, 2*np.pi, n_elements)
    r_grid, theta_grid = np.meshgrid(r_discr, theta_discr)
    x_circle = r_grid * np.cos(theta_grid) + x
    y_circle = r_grid * np.sin(theta_grid) + y
    z_circle = np.zeros_like(x_circle) + z
    return x_circle, y_circle, z_circle

def makeCylinderMeshes(x: float, y: float, z: float, radius: float, height: float, 
                 theta: float=0., phi: float=0., psi: float=0., 
                 n_elements: int=15):
    """Returns the mesh coordinates for a cylinder, composed of a wall and two 
    caps (top and bottom).
    
    Parameters
    ----------
    x, y, z : float
        Coordinates for the center of the bottom face of the cylinder.
    radius: float
        Radius of the cylinder.
    height: float
        Height of the cylinder.
    theta: float, default=0.0
        Rotation of the cylinder about the z axis (XY plane).
    phi: float, default=0.0
        Rotation of the cylinder about the y axis (XZ plane).
    psi: float, default=0.0
        Rotation of the cylinder about the x axis (YZ plane).
    n_elements: int, default=15
        Number of elements in the mesh. Lower indicates lower resolution.
    color: str, default="00DDDD"
        Color of the returned cylinder.

    Outputs
    -------
    np.ndarray: mesh in (x, y, z) pairs of the cylinder wall
    np.ndarray: mesh in (x, y, z) pairs of the bottom cylinder face (a circle)
    np.ndarray: mesh in (x, y, z) pairs of the top cylinder face (a circle)
    """
    wall_mesh = np.array(makeCylinderWall(x, y, z, radius, height, n_elements))
    bott_mesh = np.array(makeCircle(x, y, z + height, radius, n_elements))
    top_mesh = np.array(makeCircle(x, y, z, radius, n_elements))

    meshes = rotateCylinder(wall_mesh, bott_mesh, top_mesh, theta, phi, psi, x, y, z)
    return meshes

def rotateCylinder(wall, bott, top, theta: float, phi: float, psi: float, 
                   x: float=0., y: float=0., z: float=0.):
    """Rotates the cylinder meshes around x, y, and z axis, offset by x, y, and
    z coordinates."""
    if theta == 0. and phi == 0. and psi == 0.:
        return (wall, bott, top)
    T = Transform()
    T.rotate(theta, phi, psi, x, y, z)

    rotate = lambda m, T : T.transform(make2D(m))
    wall =  make3D(rotate(wall, T)[:,:3])
    bott =  make3D(rotate(bott, T)[:,:3], np.shape(bott)[1])
    top =  make3D(rotate(top, T)[:,:3], np.shape(top)[1])
    return wall, bott, top

def make2D(A):
    """Reshapes a 3D mesh matrix into a 2D matrix of ordered tuples. For example,
    a 3x6x6 matrix gets resized into a 36x3 matrix.
    """
    s = np.shape(A)
    assert len(s) == 3, f"Array is of dimension {len(s)}, not of dimension 3."
    n_coords = s[0]
    n_rows = s[1] * s[2]
    return np.resize(A, (n_coords, n_rows)).T

def make3D(A, n_rows: int=None):
    """Reshapes a 2D matrix of ordered tuples into a 3D mesh matrix. For example,
    a 24x3 matrix, with `n_rows`=6 gets resized into a 3x6x4 matrix. If `n_rows`
    is None, then the function takes the floor of the square root of the rows
    """
    s = np.shape(A)
    assert len(s) == 2, f"Array is of dimension {len(s)}, not of dimension 2."
    n_coords = s[1]
    if n_rows is None:
        n_rows = int(np.sqrt(s[0]))
        n_cols = n_rows
    else:
        n_cols = int(s[0] / n_rows)
    
    return np.resize(A.T, (n_coords, n_rows, n_cols))
    
def makeCylinder(x: float, y: float, z: float, radius: float, height: float, 
                 theta: float=0., phi: float=0., psi: float=0., 
                 n_elements: int=15, color: str='#00DDDD', opacity: float=1.0):
    """Wrapper for `makeCylinderMesh` that returns the actual figures.
    
    Parameters
    ----------
    x, y, z : float
        Coordinates for the center of the bottom face of the cylinder.
    radius: float
        Radius of the cylinder.
    height: float
        Height of the cylinder.
    theta: float, default=0.0
        Rotation of the cylinder about the z axis (XY plane).
    phi: float, default=0.0
        Rotation of the cylinder about the y axis (XZ plane).
    psi: float, default=0.0
        Rotation of the cylinder about the x axis (YZ plane).
    n_elements: int, default=15
        Number of elements in the mesh. Lower indicates lower resolution.
    color: str, default="00DDDD"
        Color of the returned cylinder.
    opacity: float, default=1.0
        Opacity of surface, with 0.0 being transparent and 1.0 being opaque

    Outputs
    -------
    go.Surface: surface of the cylinder wall
    go.Surface: surface of the bottom face (circle)
    go.Surface: surface of the top face (circle)
    """
    mesh_wall, mesh_bott, mesh_top = makeCylinderMeshes(x, y, z, radius, height, 
                                                        theta, phi, psi, n_elements)
    
    getcoord = lambda m : {'x': m[0], 'y': m[1], 'z': m[2]}
    colorscale = [[0, color], [1, color]]
    options = {
        'colorscale': colorscale,
        'opacity': opacity,
        'showscale': False
    }
    wall = go.Surface(**getcoord(mesh_wall), **options)
    bott = go.Surface(**getcoord(mesh_bott), **options)
    top = go.Surface(**getcoord(mesh_top), **options)
    return wall, bott, top

if __name__ == '__main__':
    cyl_surfaces = makeCylinder(0, 0, 0, 1, 5, theta=np.pi/4, phi=np.pi/4, n_elements=90)
    fig = go.Figure(cyl_surfaces)
    fig.show()

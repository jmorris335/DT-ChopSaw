"""
| File: animator.py 
| Info: caller script that simulates a system and plots each time-step to an animated window.
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization.
| Version History:
| - 0.1 31 Jan 2024: Initialization
"""

import matplotlib.pyplot as plt

from src.gui.blit_manager import BlitManager

def animate(entity, actions: list, rate: float=1/30):
    """
    Produces an animated matplotlib plot where each step is defined by a successive entry
    in the list actions.

    Parameters
    ----------
    entity : Any
        The entity to animate, can be any digital twin object with a `patch` member and functions
        of `set(kwargs)`, `step()`, and `updatePatches()`
    actions : list[dict]
        A list of keyword-argument pairs (as a dictionary), where each dictionary defines a list
        of values to set at a specific step of the animation. The names of the keywords come from
        the individual member(s) in the entity.
    rate : float, default=1/30
        The pause time in between each successive frame in the animation, in seconds.

    Process
    -------
    Each component needs to be independently plottable. This means that each component has the following 
    functions/member attributes:
    - `set(**kwargs)` A general setter function that where inputs can be passed to the component. The 
    keywords are specific to each components. For instance, `set(blade_omega=500)` sets the property 
    `blade_omega` to 500 rad/s if called on the blade, but would do nothing if called on the workpiece.
        Aggregate components pass the kwargs to each of their constituent members.
    - `step()` A function that updates all key parameters of the component based on any necessary inputs.
     Step should not take in any inputs, instead the function allows the component to update based on the 
     most recently provided input values. 
    - `patches` A list accessible as a member attribute that contains all `matplotlib.patches.Patch` 
    objects related to the component. This member must be initialized during the `__init__(self)` 
    initialization routine so that it is always accessible.
        Note that each object must have it's own local coordinate system to be plotted independently, even 
        when such a coordinate system does not make sense.
    - `updatePatches()` A function that updates all `matplotlib.patches.Patch` objects in `self.patches` 
    based on current state values of the component.

    The animation function is then called on any entity in the following way: `animate(entity, actions)`, 
    where entity is the object to animate and actions is a list of dictionaries providing input values for 
    each step of the animation.

    Example
    -------
    ```python
        # This animates the saw extending by 0.1 m over 2 steps
        saw = Saw()
        actions = [{'x_arm'=0.1}, {'x_arm'=0.2}]
        animate(saw, actions)
    ```
    """
    patches = entity.patches

    fig, ax = initializePlot()
    for patch in patches:
        ax.add_patch(patch)
    ax.legend(loc="lower center", ncols=2)
    bm = BlitManager(fig.canvas, patches)

    plt.show(block=False)
    plt.pause(rate)

    for action in actions:
        entity.set(**action)
        entity.step()
        entity.updatePatches()   
        
        bm.update()
        plt.pause(rate)

    plt.show(block=True)

def initializePlot():
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(-.3, .3)
    ax.set_ylim(-.2, .3)
    fig.suptitle("Workpiece and Sawblade")
    return fig, ax

def makeLinearPath(action_bounds: dict, num_steps: int=100):
    """Makes a list of dictionaries containing keyward arguments for input values for the 
    number of steps provided. Linearly interpolates the values based on the bounds given in 
    action_bounds.
    
    Parameters
    ----------
    action_bounds : dict
        A dictionary where each keyword is a input attribute of a dynamic entity, and each 
        mapped value is a list of length two with the bounds of the keyword.
    num_steps : int, default=100
        The number of steps to interpolate across, also the length of the returned list

    Example
    -------
    ```python
        action_bounds = {'x_arm' : [0, 0.2], 'theta_arm' : [90, 0]}
        actions = makeLinearPath(action_bounds, 50)
        saw = Saw()
        animate(saw, actions)
    ```
    """
    actions = list()
    for i in range(num_steps):
        action = dict()
        for key, val in action_bounds.items():
            action[key] = val[0] + (val[1] - val[0]) / num_steps * i
        actions.append(action)
    return actions

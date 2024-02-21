"""
| File: plotter.py 
| Info: caller script that provides functionality for plotting and animating of a system
| Author: John Morris, jhmrrs@clemson.edu  
| Organization: Product Lifecycle Management Center at Clemson University, plmcenter@clemson.edu  
| Permission: Copyright (C) 2023, John Morris. All rights reserved. Should not be reproduced, edited, sourced, or utilized without written permission from the author or organization.
| Version History:
| - 0.1 31 Jan 2024: Initialization
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from src.gui.blit_manager import BlitManager
from src.objects.twin import Twin
from src.db.logger import Logger

def plotStatic(entities: list, block: bool=True):
    """Plots the entity without simulating the entity. Similar to calling
    `animate(entity, actions=[])`, but can take a list of entities rather than an aggregate
    model."""
    if not hasattr(entities, '__len__'): entities = [entities]
    patches = list()
    for entity in entities:
        entity.updatePatches()
        patches.extend(entity.patches)

    fig, ax = plt.subplots()
    for patch in patches:
        ax.add_patch(patch)
    ax.legend(loc="lower center", ncols=2)
    ax.axis('equal')

    plt.show(block=block)

def animateWithData(entity: Twin, actions: list, logger: Logger, y_names: list, rate: float=1/30):
    """
    Animates the entity alongside an evolving data_chart versus time.
    """
    entity.updatePatches()
    artists = entity.patches

    plt.rcParams["font.family"] = "Helvetica"
    fig, axs = plt.subplots(1, ncols=2, figsize=(10, 5))
    plt.gcf().text(0.75, 0.0, "© 2024 PLM Center at Clemson University", fontsize=8, color='#A4A4A4')
    fig.patch.set_facecolor('#F3F3F3')
    fig, axs[0] = configurePlot(fig, axs[0])

    dt = 0.01 if logger is None else logger.entity.time_step
    axs[1].set_box_aspect(1)
    axs[1].set_xlim(0, dt*len(actions))
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel("Values")
    axs[1].set_title("Simulated Data")
    axs[1].grid()

    # Setup DT
    for artist in artists:
        axs[0].add_patch(artist)
    box = axs[0].get_position()
    axs[0].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, ncol=2)

    # Setup data plot
    t = list()
    y = list()
    lines = list()
    def formatLabel(s: str):
        s = s.replace('_', ' ')
        s = s.capitalize()
        return s
    for i in range(len(y_names)):
        t.append([])
        y.append([])
        line, = axs[1].plot(t[i], y[i], lw=2, animated=True, label=formatLabel(y_names[i]))
        lines.append(line)
    artists.extend(lines)
    box = axs[1].get_position()
    axs[1].set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    axs[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, ncol=2)

    bm = BlitManager(fig.canvas, artists)
    plt.show(block=False)
    plt.pause(rate)

    for action in actions:
        entity.set(**action)
        entity.step()
        entity.updatePatches()

        for i in range(len(y_names)):
            data = logger.getLatestEntry(y_names[i])
            y[i].append(data[logger.col_idx('value')])
            t[i].append(data[logger.col_idx('time')])
            lines[i].set_ydata(y[i])
            lines[i].set_xdata(t[i])
            y_lim = axs[1].get_ylim()
            if y[i][-1] < y_lim[0]:
                max_y = max(max(a) for a in y if len(a) > 0)
                min_y = min(min(a) for a in y if len(a) > 0)
                buffer = abs(max_y - min_y) * 0.1
                axs[1].set_ylim(min_y - buffer, y_lim[1])
                plt.show(block=False)
            elif y[i][-1] > y_lim[1]:
                max_y = max(max(a) for a in y if len(a) > 0)
                min_y = min(min(a) for a in y if len(a) > 0)
                buffer = abs(max_y - min_y) * 0.1
                axs[1].set_ylim(y_lim[0], max_y + buffer)
                plt.show(block=False)
        bm.update()
        plt.pause(rate)

    plt.show(block=True)


def animate(entity: Twin, actions: list, rate: float=1/30, fig: Figure=None, ax :Axes=None, block: bool=True):
    """
    Simulates the entity and outputs the system to an animated matplotlib plot where each 
    step is defined by a successive entry in the list actions.

    Parameters
    ----------
    entity : Twin
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
    entity.updatePatches()
    patches = entity.patches

    fig, ax = configurePlot(fig, ax)
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

    plt.show(block=block)

def configurePlot(fig: Figure=None, ax: Axes=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_box_aspect(1)
    ax.set_xlim(-.3, .3)
    ax.set_ylim(-.2, .3)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle("Workpiece and Sawblade")
    ax.set_title("Simulation")
    return fig, ax

def configurePlots(fig: Figure=None, axs: list=None):
    if fig is None or axs is None:
        fig, axs = plt.subplots(ncols=2)
    fig, axs[0] = configurePlot(fig, axs[0])
    return fig, axs

def configureGraph(fig: Figure=None, ax: Axes=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots()
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
    if num_steps == 1: num_steps += 1
    actions = list()
    for i in range(num_steps):
        action = dict()
        for key, val in action_bounds.items():
            action[key] = val[0] + (val[1] - val[0]) / (num_steps-1) * i
        actions.append(action)
    return actions

def plotData(logger: Logger, x_name: str, y_name: str, ax: Axes=None):
    """Retrieves the data from the DB with the provided logger and plots it to the
    passed Axes object."""
    if ax is None:
        fig, ax = configureGraph(ax= ax)
    if x_name == 'time':
        rows = logger.getAllRows(label=y_name)
        x = [row[logger.col_idx('time')] for row in rows]
        y = [row[logger.col_idx('value')] for row in rows]
    else:
        x = logger.getAllValues(x_name)
        y = logger.getAllValues(y_name)
    
    ax.plot(x, y, lw=2, color='k')
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    plt.show()

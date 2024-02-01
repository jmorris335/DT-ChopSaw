# Digital Twin Philosophy
## Dynamics

## Animation
Each component needs to be independently plottable. This means that each component has the following functions/member attributes:
- `set(**kwargs)` A general setter function that where inputs can be passed to the component. The keywords are specific to each components. For instance, `set(blade_omega=500)` sets the property `blade_omega` to 500 rad/s if called on the blade, but would do nothing if called on the workpiece.
    
    Aggregate components pass the kwargs to each of their constituent members.

- `step()` A function that updates all key parameters of the component based on any necessary inputs. Step should not take in any inputs, instead the function allows the component to update based on the most recently provided input values. 
- `patches` A list accessible as a member attribute that contains all `matplotlib.patches.Patch` objects related to the component. This member must be initialized during the `__init__(self)` initialization routine so that it is always accessible.

    Note that each object must have it's own local coordinate system to be plotted independently, even when such a coordinate system does not make sense.

- `updatePatches()` A function that updates all `matplotlib.patches.Patch` objects in `self.patches` based on current state values of the component.

The animation function is then called on any entity in the following way: `animate(entity, actions)`, where entity is the object to animate and actions is a list of dictionaries providing input values for each step of the animation.

Example:
```python
# This animates the saw extending by 0.1 m over 2 steps
saw = Saw()
actions = [{'x_arm'=0.1}, {'x_arm'=0.2}]
animate(saw, actions)
```
'''
General supporting functions for all objects or methods
'''

import yaml
from types import SimpleNamespace

def findDefault(default_val, key, kwargs: dict):
    ''' Returns default_val if the key is not found in the dict. Useful for setting default dictionary args
    '''
    return default_val if key not in kwargs else kwargs[key]

def loopSlice(l, i, j):
    """
    Treating l (array_like) as a sequential deque, returns all elements of l that lie
    between i and j (allowing for wrap around indexing). Note that indexes follow python slicing
    rules, so loopSlice[l, i, j] includes l[i] but not l[j].
    
    Example
    -------
    ```
    l = [1, 2, 3, 4, 5]
    out = loopSlice(l, 3, 1)
    print(out) #returns [4, 5, 1]
    out = loopSlice(l, 4, 3)
    print(out) #returns [5, 1, 2, 3]
    ```
    """
    if i < j: return l[i:j]
    if i == j: return l[:]
    return l[i:] + l[:j]

def findDepth(l, level=0) -> int:
    """Returns the depth of the iterable object l. For example, if l = [[1, 2], [3, 4]],
    then the function would return a depth of 2. Returns the first level containing a 
    non-iterable item, i.e. [1, [2, 3], [4, 5]] has a depth of 1."""
    if not (isinstance(l, list) or isinstance(l, tuple)): 
        return level
    level += 1
    min_level = min([findDepth(a, level) for a in l])
    return min_level

def shiftIndices(indices: list, index: int=0, shiftPos: bool=True):
    """Adjusts a list of indices (to another array_like object) to reflect inserts (shiftPos=
    True) or deletions (shiftPos=False) at index in the mapped array."""
    for i in range(len(indices)):
        if indices[i] >= index:
            indices[i] += 1 if shiftPos else -1

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

'''
General supporting functions for all objects or methods
'''

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
    if i <= j: return l[i:j]
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

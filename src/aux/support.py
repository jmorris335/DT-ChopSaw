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
'''
General supporting functions for all objects or methods
'''

def findDefault(default_val, key, kwargs: dict):
    ''' Returns default_val if the key is not found in the dict. Useful for setting default dictionary args
    '''
    return default_val if key not in kwargs else kwargs[key]
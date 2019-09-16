import dill

def dictToAttributes(cls, params):

    """
    Assign member attributes to class via dictionary.
    Keys are variable values and value is variable's value
    """
    
    # Assign Attributes
    for key, value in params.items():

        setattr(cls, '_' + key, value)

def saveDict(save_dict=None, save_path=None):

    assert dict is not None
    assert save_path is not None
    
    # copy self.__dict__ to save_dict
    save_dict_copy = dict()

    # remove leading underscores from save_dict
    for key in list(save_dict.keys()):

        save_dict_copy[key[1:]] = save_dict[key]

    # write save_dict to disk
    with open(save_path, 'wb') as handle:

        dill.dump(save_dict_copy, handle)

def dictToAttributes(cls, params):

    """
    Assign member attributes to class via dictionary.
    Keys are variable values and value is variable's value
    """
    
    # Assign Attributes
    for key, value in params.items():

        setattr(cls, '_' + key, value)

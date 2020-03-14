import numpy as np
import dill
from pdb import set_trace as st

def dictToAttributes(cls, params):

    """
    Assign member attributes to class via dictionary.
    Keys are variable values and value is variable's value
    """

    # Assign Attributes
    for key, value in params.items():

        setattr(cls, '_' + key, value)

def saveDict(save_dict=None, save_path=None):

    assert save_dict is not None
    assert save_path is not None

    # write save_dict to disk
    with open(save_path, 'wb') as handle:

        dill.dump(save_dict, handle)

def saveAttrDict(save_dict=None, save_path=None):

    assert save_dict is not None
    assert save_path is not None

    # copy self.__dict__ to save_dict
    save_dict_copy = dict()

    # remove leading underscores from save_dict
    for key in list(save_dict.keys()):

        save_dict_copy[key[1:]] = save_dict[key]

    # write save_dict to disk
    with open(save_path, 'wb') as handle:

        dill.dump(save_dict_copy, handle)

def loadDict(load_path=None):

    assert load_path is not None

    # pickle file
    try:

        # write save_dict to disk
        with open(load_path, 'rb') as handle:

            return dill.load(handle)

    # npy file
    except:

        return np.load(load_path,allow_pickle=True).item()

def generateMask(data=None,mask_percent=None):

    assert data is not None
    assert mask_percent is not None

    for x in data:

        mask_indices = np.random.choice(np.arange(x.shape[0]-1), replace=False, size=round(x.shape[0]*mask_percent))

        # randomly pad additional missing data before and after each mask index with 25% probability
        for mask_index in mask_indices:

            bin_index = np.random.choice([0,1],p=[0.75,0.25])

            if bin_index == 1:

                mask_indices = np.append(mask_indices,mask_index+bin_index)

                # ensure negative index does not occur
                if mask_index != 0:

                    mask_indices = np.append(mask_indices,mask_index-bin_index)

    return mask_indices

# modules
import sys
import numpy as np

def bimodal( loc1, scale1, loc2, scale2, size ):

    # list to hold samples
    noise = list()

    # array of 0/1's
    binary_choice = np.random.randint(low=0, high=2, size=np.prod(size))

    # loop over all binary choices
    for choice in binary_choice:

        # select location and scale
        if choice == 0:

            loc, scale = loc1, scale1

        else:

            loc, scale = loc2, scale2

        # sample from given normal distribution and append to data list
        noise.append(np.random.normal(loc=loc, scale=scale, size=(1)))

    # list to np array
    noise = np.asarray(noise).reshape(size)

    # center noise
    noise -= np.mean(noise)

    return noise

###############################################################################
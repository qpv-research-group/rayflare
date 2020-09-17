from collections import OrderedDict
import numpy as np

class State(OrderedDict):
    """ This class defines a convent way of expanding the attributes of an object, usually fixed during the definition of
    the class. In this case, the class is just a dictionary - a special type of it - and attributes are expanded by
    adding new keys to it.
    """
    def __getattr__(self, name):
        # print ("***", name)
        if name in ["_OrderedDict__root", "_OrderedDict__map"]:
            return OrderedDict.__getattribute__(self, name)

        if name in self:
            return self[name]
        else:
            raise KeyError("The state object does not have an entry for the key '%s'." % (name,))

    def __setattr__(self, name, value):
        if name in ["_OrderedDict__root", "_OrderedDict__map", "_OrderedDict__hardroot"]:
            return OrderedDict.__setattr__(self, name, value)

        self[name] = value


default_options = State()

default_options.n_theta_bins = 100
default_options.c_azimuth = 0.25
default_options.nm_spacing = 1
default_options.pol = 'u'
default_options.wavelengths = np.linspace(300, 1500, 100)
default_options.theta_in = 1e-6 # TODO: be able to just set theta_in = 0
default_options.phi_in = 1e-6
default_options.I_thresh = 1e-5
default_options.lookuptable_angles = 300
default_options.n_rays = 1e5
default_options.random_angles = False
default_options.nx = 5
default_options.ny = 5
default_options.parallel = True
default_options.n_jobs = -1
default_options.phi_symmetry = np.pi/4
default_options.only_incidence_angle = True

from rayflare.state import State
import numpy as np

class default_options(State):
    def __init__(self):
        self.wavelengths = np.linspace(300, 1800, 300)*1e-9
        self.phi_in = 0
        self.theta_in = 0
        self.phi_symmetry = np.pi/4
        self.n_theta_bins = 50
        self.I_thresh = 1e-2
        self.depth_spacing = 1
        self.parallel = True
        self.pol = 'u'
        self.n_jobs = -1
        self.c_azimuth = 0.25
        self.only_incidence_angle = True
        
        # RCWA options
        self.A_per_order = False
        self.S4_options = dict(LatticeTruncation='Circular',
                                DiscretizedEpsilon=False,
                                DiscretizationResolution=8,
                                PolarizationDecomposition=False,
                                PolarizationBasis='Default',
                                LanczosSmoothing=False,
                                SubpixelSmoothing=False,
                                ConserveMemory=False,
                                WeismannFormulation=False)
        
        
        # Ray-tracing options
        self.nx = 10
        self.ny = 10
        self.random_ray_position = False
        self.avoid_edges = False
        self.randomize_surface = False
        self.random_ray_angles = False
        
        # TMM options
        self.lookuptable_angles = 300
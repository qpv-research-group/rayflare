"""FMMAX example: patterned GaAs above a finite silver layer.

Install the optional backend first with ``pip install "rayflare[fmmax]"``.
"""

import matplotlib.pyplot as plt
import numpy as np

from solcore import material, si
from solcore.structure import Layer

from rayflare.options import default_options
from rayflare.rigorous_coupled_wave_analysis import rcwa_structure


air = material("Air")()
gaas = material("GaAs")()
silver = material("Ag")()

pitch = 400  # nm: RCWA geometry uses nm
layers = [
    Layer(
        si(300, "nm"),
        gaas,
        geometry=[{"type": "circle", "mat": air, "center": (0, 0), "radius": 100}],
    ),
    Layer(si(50, "nm"), silver),
]

options = default_options()
options.RCWA_method = "FMMAX"
options.wavelength = np.linspace(500, 900, 9) * 1e-9
options.theta_in = np.deg2rad(15)
options.phi_in = 0
options.pol = "u"
options.orders = 25
options.parallel = False
options.FMMAX_options.update(grid_spacing=5.0, formulation="fft")

solver = rcwa_structure(
    layers,
    size=((pitch, 0), (0, pitch)),
    options=options,
    incidence=air,
    transmission=air,
)
result = solver.calculate(options)

# The two columns correspond to the two finite layers above. The air holes are
# nonabsorbing, so absorption in layer 0 is GaAs absorption; layer 1 is Ag loss.
gaas_absorption = result["A_per_layer"][:, 0]
silver_absorption = result["A_per_layer"][:, 1]

print("wavelength (nm)   R       T       A_GaAs  A_Ag")
for wavelength, reflection, transmission, a_gaas, a_ag in zip(
    options.wavelength * 1e9,
    result["R"],
    result["T"],
    gaas_absorption,
    silver_absorption,
):
    print(
        f"{wavelength:8.1f}       {reflection:.4f}  {transmission:.4f}  "
        f"{a_gaas:.4f}  {a_ag:.4f}"
    )

plt.plot(options.wavelength * 1e9, gaas_absorption, "o-", label="GaAs absorption")
plt.plot(options.wavelength * 1e9, silver_absorption, "s-", label="Ag absorption")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbed fraction")
plt.legend()
plt.tight_layout()
plt.show()

# For specified incident spectral irradiance I_lambda [W m^-2 m^-1], the
# absorbed-photon rate in GaAs is
#
#   Phi_abs(lambda) = A_GaAs(lambda) I_lambda(lambda) / (h c / lambda).
#
# Integrating Phi_abs over wavelength gives absorbed photons m^-2 s^-1. Under
# an explicitly stated unity internal collection efficiency this can be used as
# a photogeneration/current upper bound. A depth profile from calculate_profile
# gives G(z, lambda) after converting its nm^-1 unit to m^-1. Silver absorption
# is parasitic optical heating here: it is not electron-hole photogeneration in
# GaAs and this electromagnetic calculation does not predict photoemission EQE.

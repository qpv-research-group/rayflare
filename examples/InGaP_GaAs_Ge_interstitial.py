from rayflare.rigorous_coupled_wave_analysis import rcwa_structure
from rayflare.transfer_matrix_method import tmm_structure
from solcore import material, si
from solcore.structure import Layer
from rayflare.options import default_options
import numpy as np
import matplotlib.pyplot as plt
from solcore.light_source import LightSource
from solcore.constants import q
import seaborn as sns
from matplotlib import rc

# Paper: https://doi.org/10.1016/j.solmat.2016.09.005

rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"], "size": 15})

pal = sns.color_palette("husl", 3)

InAlP = material("AlInP")(Al=0.5)
GaAs_pn_junction = material("GaAs")()
InGaP = material("GaInP")(In=0.5)
Ge = material("Ge")()
Ag = material("Ag")()
Au = material("Au")()
SiN = material("Si3N4")()
Air = material("Air")()
Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
MgF2 = material("MgF2")()  # MgF2 (SOPRA database)
epoxy = material("BK7")()
TiO2 = material("TiO2")()

ARC = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5)]

centre_wavelength = 750
d_MgF2 = centre_wavelength / (4 * MgF2.n(centre_wavelength * 1e-9))
d_TiO2 = centre_wavelength / (4 * TiO2.n(centre_wavelength * 1e-9))

DBR = 25 * [Layer(d_MgF2 * 1e-9, MgF2), Layer(d_TiO2 * 1e-9, TiO2)]

wavelengths = np.linspace(300, 1850, 300) * 1e-9

photon_flux = LightSource(
    source_type="standard", version="AM0", x=wavelengths, output_units="photon_flux_per_m"
).spectrum(wavelengths)[1]

grating_circles = [{"type": "circle", "mat": InGaP, "center": (0, 0), "radius": 185}]

x = 460

size = ((x, 0), (0, x))

options = default_options()
options.wavelength = wavelengths
options.orders = 60

layers = (
    ARC
    + [
        Layer(material=InAlP, width=si("30nm")),
        Layer(material=InGaP, width=si("400nm")),
        Layer(material=GaAs_pn_junction, width=si("700nm")),
        Layer(material=epoxy, width=si("230nm"), geometry=grating_circles),
        Layer(material=epoxy, width=si("800nm")),
    ]
    + DBR
)


S4_setup = rcwa_structure(layers, size=size, options=options, incidence=Air, transmission=Ge)
RAT = S4_setup.calculate(options)

J_InGaP = q * np.trapz(RAT["A_per_layer"][:, 3] * photon_flux, x=wavelengths) / 10
J_GaAs = q * np.trapz(RAT["A_per_layer"][:, 4] * photon_flux, x=wavelengths) / 10
J_Ge = q * np.trapz(RAT["T"] * photon_flux, x=wavelengths) / 10

print("Jsc InGaP =", J_InGaP, "mA/cm2")
print("Jsc GaAs =", J_GaAs, "mA/cm2")
print("Jsc Ge =", J_Ge, "mA/cm2")

plt.figure(figsize=(10, 5))
plt.plot(wavelengths * 1e9, RAT["R"], "--k", label="Reflectance")
plt.plot(wavelengths * 1e9, RAT["A_per_layer"][:, 3], label="InGaP absorption", color=pal[0])
plt.plot(wavelengths * 1e9, RAT["A_per_layer"][:, 4], label="GaAs absorption", color=pal[1])
plt.plot(wavelengths * 1e9, RAT["T"], label="Transmission into Ge", color=pal[2])
plt.legend(loc=(1.05, 0.6))
plt.xlim(300, 1850)
plt.ylim(0, 1)
plt.text(500, 0.5, "InGaP: \n" + str(round(J_InGaP, 1)) + r" mA/cm$^2$", horizontalalignment="center")
plt.text(760, 0.4, "GaAs: \n" + str(round(J_GaAs, 1)) + r" mA/cm$^2$", horizontalalignment="center")
plt.text(1500, 0.85, "Ge: \n" + str(round(J_Ge, 1)) + r" mA/cm$^2$", horizontalalignment="center")
plt.xlabel("Wavelegnth (nm)")
plt.ylabel("R / A / T")
plt.grid()
plt.tight_layout()
plt.show()


DBR_str = tmm_structure(DBR, incidence=epoxy, transmission=Ge)

RAT_DBR = DBR_str.calculate(options)

plt.figure()
plt.plot(wavelengths * 1e9, RAT_DBR["R"], "--")
plt.show()

layers = ARC + [
    Layer(material=InAlP, width=si("30nm")),
    Layer(material=InGaP, width=si("400nm")),
    Layer(material=GaAs_pn_junction, width=si("3500nm")),
]


S4_setup = rcwa_structure(layers, size=size, options=options, incidence=Air, transmission=Ge)
RAT_ref1 = S4_setup.calculate(options)

layers = ARC + [
    Layer(material=InAlP, width=si("30nm")),
    Layer(material=InGaP, width=si("400nm")),
    Layer(material=GaAs_pn_junction, width=si("700nm")),
    Layer(material=epoxy, width=si("230nm"), geometry=grating_circles),
    Layer(material=epoxy, width=si("800nm")),
]


S4_setup = rcwa_structure(layers, size=size, options=options, incidence=Air, transmission=Ge)
RAT_noDBR = S4_setup.calculate(options)

layers = (
    ARC
    + [
        Layer(material=InAlP, width=si("30nm")),
        Layer(material=InGaP, width=si("400nm")),
        Layer(material=GaAs_pn_junction, width=si("700nm")),
        # Layer(material=epoxy, width=si("230nm"), geometry=grating_circles),
        # Layer(material=epoxy, width=si("800nm"))
    ]
    + DBR
)


tmm_setup = tmm_structure(layers, incidence=Air, transmission=Ge)
RAT_nograting = tmm_setup.calculate(options)

layers = ARC + [
    Layer(material=InAlP, width=si("30nm")),
    Layer(material=InGaP, width=si("400nm")),
    Layer(material=GaAs_pn_junction, width=si("700nm")),
    # Layer(material=epoxy, width=si("230nm"), geometry=grating_circles),
    # Layer(material=epoxy, width=si("800nm"))
]

tmm_setup = tmm_structure(layers, incidence=Air, transmission=Ge)
RAT_700 = tmm_setup.calculate(options)

plt.figure(figsize=(9, 5))

plt.plot(wavelengths * 1e9, RAT_700["A_per_layer"][:, 4], "-k", label="700 nm GaAs")
plt.plot(wavelengths * 1e9, RAT_nograting["A_per_layer"][:, 4], label="700 nm GaAs + DBR", color=pal[0])
plt.plot(wavelengths * 1e9, RAT_noDBR["A_per_layer"][:, 4], label="700 nm GaAs + grating", color=pal[1])
plt.plot(wavelengths * 1e9, RAT["A_per_layer"][:, 4], label="700 nm GaAs + grating + DBR", color=pal[2])
plt.plot(wavelengths * 1e9, RAT_ref1["A_per_layer"][:, 4], "--k", label="3500 nm GaAs")

plt.xlabel("Wavelength (nm)")
plt.ylabel("Absorbed in GaAs junction")
plt.xlim(400, 900)
plt.ylim(0, 1)

plt.legend(loc=(1.05, 0.6))
plt.grid()
plt.tight_layout()
plt.show()

labels = ["700 nm GaAs", "700 nm GaAs + grating", "700 nm GaAs + DBR", "700 nm GaAs + grating + DBR", "3500 nm GaAs"]

A_list = [
    RAT_700["A_per_layer"][:, 4],
    RAT_nograting["A_per_layer"][:, 4],
    RAT_noDBR["A_per_layer"][:, 4],
    RAT["A_per_layer"][:, 4],
    RAT_ref1["A_per_layer"][:, 4],
]

T_list = [RAT_700["T"], RAT_nograting["T"], RAT_noDBR["T"], RAT["T"], RAT_ref1["T"]]

Jscs = np.zeros(len(A_list))
Jscs_Ge = np.zeros(len(A_list))

for i, A in enumerate(A_list):
    Jscs[i] = q * np.trapz(A * photon_flux, wavelengths) / 10
    Jscs_Ge[i] = q * np.trapz(photon_flux * T_list[i], wavelengths) / 10

plt.figure(figsize=(5, 5))
plt.plot(labels, Jscs, "o-", fillstyle="none", color=pal[0], label="GaAs", markersize=8)
plt.plot(labels, Jscs_Ge, "x-", color=pal[1], label="Ge", markersize=8)
plt.ylabel(r"J$_{sc, max}$ (mA/cm$^2$)")
plt.xticks(rotation=45, ha="right")
plt.legend()
plt.tight_layout()
plt.show()

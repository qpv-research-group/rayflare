# Copyright (C) 2026 Phoebe Pearce and contributors
#
# This file is part of RayFlare and is released under the GNU Lesser General
# Public License (LGPL), version 3. Please see LICENSE.txt.

"""Optional FMMAX implementation for RayFlare's RCWA interface.

RayFlare describes RCWA structures in nanometres and uses refractive indices
``n + 1j * k``. FMMAX is run in the same length unit, with permittivity
``(n + 1j * k) ** 2``. Both conventions correspond to fields evolving as
``exp(-1j * omega * t)``.

FMMAX and JAX are deliberately imported only when this backend is selected.
"""

from importlib import metadata

import numpy as np


FMMAX_PIN = "1.7.1"

DEFAULT_OPTIONS = {
    "formulation": "fft",
    "truncation": "circular",
    "grid_spacing": 5.0,
    "grid_shape": None,
    "force_x64_solve": False,
    "max_grid_points": 1_000_000,
}


def _imports():
    """Import FMMAX and JAX, with an actionable optional-dependency error."""
    try:
        import jax.numpy as jnp
        from fmmax import basis, fields, fmm, scattering
    except ImportError as err:
        raise ImportError(
            "The FMMAX RCWA backend requires the optional FMMAX/JAX "
            f"dependencies. Install them with `pip install 'rayflare[fmmax]'` "
            f"or `pip install fmmax=={FMMAX_PIN}`."
        ) from err
    return jnp, basis, fields, fmm, scattering


def version():
    """Return the installed FMMAX version, importing no JAX modules."""
    try:
        return metadata.version("fmmax")
    except metadata.PackageNotFoundError as err:
        raise ImportError(
            "The FMMAX RCWA backend is selected, but FMMAX is not installed. "
            "Install it with `pip install 'rayflare[fmmax]'`."
        ) from err


def options(user_options=None):
    """Validate and merge FMMAX-specific options."""
    result = DEFAULT_OPTIONS.copy()
    user_options = {} if user_options is None else dict(user_options)
    unknown = sorted(set(user_options) - set(result))
    if unknown:
        raise ValueError(f"Unsupported FMMAX_options: {', '.join(unknown)}")
    result.update(user_options)

    if result["grid_spacing"] <= 0:
        raise ValueError("FMMAX_options['grid_spacing'] must be positive (in nm).")
    if result["grid_shape"] is not None:
        shape = tuple(result["grid_shape"])
        if len(shape) != 2 or any(int(x) != x or x < 1 for x in shape):
            raise ValueError(
                "FMMAX_options['grid_shape'] must contain two positive integers."
            )
        result["grid_shape"] = tuple(int(x) for x in shape)
    if result["max_grid_points"] < 1:
        raise ValueError("FMMAX_options['max_grid_points'] must be positive.")
    return result


def _enum_from_value(enum_class, value, option_name):
    if isinstance(value, enum_class):
        return value
    value = str(value).lower()
    for member in enum_class:
        if member.value == value:
            return member
    supported = ", ".join(member.value for member in enum_class)
    raise ValueError(
        f"Unsupported FMMAX {option_name} {value!r}; choose from {supported}."
    )


def _grid_shape(lattice, expansion, backend_options, basis):
    if backend_options["grid_shape"] is None:
        shape = np.ceil(
            np.linalg.norm(lattice, axis=1) / backend_options["grid_spacing"]
        ).astype(int)
        shape = np.maximum(shape, 1)
    else:
        shape = np.asarray(backend_options["grid_shape"], dtype=int)

    minimum = np.asarray(basis.min_array_shape_for_expansion(expansion), dtype=int)
    if backend_options["grid_shape"] is not None and np.any(shape < minimum):
        raise ValueError(
            "FMMAX_options['grid_shape'] is too small for the Fourier expansion: "
            f"minimum {tuple(minimum)}, got {tuple(shape)}."
        )
    shape = np.maximum(shape, minimum)
    if int(np.prod(shape)) > backend_options["max_grid_points"]:
        raise ValueError(
            f"FMMAX geometry grid {tuple(shape)} exceeds max_grid_points="
            f"{backend_options['max_grid_points']}. Increase grid_spacing or the limit."
        )
    return tuple(int(x) for x in shape)


def _cell_coordinates(lattice, shape):
    i, j = np.meshgrid(
        (np.arange(shape[0]) + 0.5) / shape[0],
        (np.arange(shape[1]) + 0.5) / shape[1],
        indexing="ij",
    )
    x = i * lattice[0, 0] + j * lattice[1, 0]
    y = i * lattice[0, 1] + j * lattice[1, 1]
    return x, y


def _points_in_polygon(x, y, vertices):
    inside = np.zeros(x.shape, dtype=bool)
    xj, yj = vertices[-1]
    for xi, yi in vertices:
        crossing = (yi > y) != (yj > y)
        x_crossing = (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        inside ^= crossing & (x < x_crossing)
        xj, yj = xi, yi
    return inside


def _shape_mask(shape, x, y, lattice):
    shape_type = str(shape.get("type", "")).lower()
    if shape_type not in {"circle", "ellipse", "rectangle", "polygon"}:
        raise NotImplementedError(
            "The FMMAX backend supports circle, ellipse, rectangle, and polygon "
            f"geometry; got {shape_type!r}."
        )

    center = np.asarray(shape.get("center", (0.0, 0.0)), dtype=float)
    if center.shape != (2,):
        raise ValueError("A geometry center must contain two coordinates in nm.")
    fractional_center = center @ np.linalg.inv(lattice)
    center = np.mod(fractional_center, 1) @ lattice
    angle = np.deg2rad(float(shape.get("angle", 0.0)))
    cosine, sine = np.cos(angle), np.sin(angle)
    mask = np.zeros(x.shape, dtype=bool)
    maximum_extent = min(np.linalg.norm(lattice, axis=1))

    # RayFlare/S4 geometry is periodically continued and commonly has a feature
    # centred at (0, 0). Check neighbouring images so such features wrap across
    # the FMMAX cell, whose origin is one of the cell vertices.
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            image_center = center + a * lattice[0] + b * lattice[1]
            dx = x - image_center[0]
            dy = y - image_center[1]
            xr = cosine * dx + sine * dy
            yr = -sine * dx + cosine * dy

            if shape_type == "circle":
                radius = float(shape["radius"])
                if radius <= 0:
                    raise ValueError("Circle radius must be positive.")
                feature_extent = radius
                image_mask = xr**2 + yr**2 <= radius**2
            elif shape_type in {"ellipse", "rectangle"}:
                halfwidths = np.asarray(shape["halfwidths"], dtype=float)
                if halfwidths.shape != (2,) or np.any(halfwidths <= 0):
                    raise ValueError(
                        "Shape halfwidths must contain two positive values."
                    )
                feature_extent = np.linalg.norm(halfwidths)
                if shape_type == "ellipse":
                    image_mask = (xr / halfwidths[0]) ** 2 + (
                        yr / halfwidths[1]
                    ) ** 2 <= 1
                else:
                    image_mask = (np.abs(xr) <= halfwidths[0]) & (
                        np.abs(yr) <= halfwidths[1]
                    )
            else:
                vertices = np.asarray(shape["vertices"], dtype=float)
                if (
                    vertices.ndim != 2
                    or vertices.shape[0] < 3
                    or vertices.shape[1] != 2
                ):
                    raise ValueError(
                        "Polygon vertices must have shape (n, 2), with n >= 3."
                    )
                feature_extent = np.max(np.linalg.norm(vertices, axis=1))
                # Vertices are relative to the centre and rotate with the shape.
                image_mask = _points_in_polygon(xr, yr, vertices)
            if feature_extent > maximum_extent:
                raise NotImplementedError(
                    "FMMAX periodic features must fit within one primitive-cell span."
                )
            mask |= image_mask
    return mask


def _permittivity_grids(
    geom_list,
    layer_permittivities,
    shape_permittivities,
    shape_names,
    lattice,
    shape,
):
    x, y = _cell_coordinates(lattice, shape)
    shape_lookup = dict(zip(shape_names, shape_permittivities))
    grids = []
    for layer_index, (geometry, epsilon) in enumerate(
        zip(geom_list, layer_permittivities)
    ):
        if not geometry:
            grids.append(np.asarray([[epsilon]], dtype=complex))
            continue
        grid = np.full(shape, epsilon, dtype=complex)
        for feature in geometry:
            material = feature.get("mat")
            if material not in shape_lookup:
                raise ValueError(
                    f"No optical constants were supplied for geometry material {material!r} "
                    f"in layer {layer_index}."
                )
            grid[_shape_mask(feature, x, y, lattice)] = shape_lookup[material]
        grids.append(grid)
    return grids


def _incident_amplitudes(
    layer_solve_result, polar_angle, azimuthal_angle, pol, fields, jnp
):
    num_terms = layer_solve_result.expansion.num_terms
    conversion = np.asarray(fields.field_conversion_matrix(layer_solve_result))
    forward_conversion = conversion[:, : 2 * num_terms]
    # field_conversion_matrix rows are (-Ey, Ex, Hx, Hy).
    electric_conversion = np.vstack(
        [
            forward_conversion[num_terms : 2 * num_terms],
            -forward_conversion[:num_terms],
        ]
    )

    s_vector = np.asarray([-np.sin(azimuthal_angle), np.cos(azimuthal_angle)])
    p_vector = np.cos(polar_angle) * np.asarray(
        [np.cos(azimuthal_angle), np.sin(azimuthal_angle)]
    )

    if isinstance(pol, str):
        pol = pol.lower()
        if pol == "s":
            vectors, weights = [s_vector], np.asarray([1.0])
        elif pol == "p":
            vectors, weights = [p_vector], np.asarray([1.0])
        elif pol == "u":
            # Unpolarized illumination is an incoherent equal-intensity average.
            vectors, weights = [s_vector, p_vector], np.asarray([0.5, 0.5])
        else:
            raise ValueError("Polarization must be 's', 'p', 'u', or an (s, p) tuple.")
    else:
        components = np.asarray(pol, dtype=complex)
        if components.shape != (2,):
            raise ValueError("A polarization tuple must contain the (s, p) amplitudes.")
        vectors = [components[0] * s_vector + components[1] * p_vector]
        weights = np.asarray([1.0])

    targets = np.zeros((2 * num_terms, len(vectors)), dtype=complex)
    for index, vector in enumerate(vectors):
        targets[0, index] = vector[0]
        targets[num_terms, index] = vector[1]
    amplitudes = np.linalg.solve(electric_conversion, targets)
    return jnp.asarray(amplitudes), weights


def _order_flux(amplitude, layer_solve_result, direction, fields, jnp):
    zero = jnp.zeros_like(amplitude)
    if direction == "forward":
        flux = fields.amplitude_poynting_flux(amplitude, zero, layer_solve_result)[0]
    else:
        flux = -fields.amplitude_poynting_flux(zero, amplitude, layer_solve_result)[1]
    num_terms = layer_solve_result.expansion.num_terms
    return np.asarray(flux[:num_terms] + flux[num_terms:])


def _weighted_normalized(values, incident_flux, weights):
    values = np.asarray(values)
    incident_flux = np.asarray(incident_flux)
    normalized = values / incident_flux[np.newaxis, :]
    return np.sum(normalized * weights[np.newaxis, :], axis=-1)


def solve(
    wavelength,
    geom_list,
    layer_permittivities,
    shape_permittivities,
    shape_names,
    pol,
    theta,
    phi,
    widths,
    size,
    orders,
    user_options=None,
    profile_depths=None,
):
    """Solve one RayFlare RCWA wavelength with FMMAX.

    Length quantities are in nm, and ``theta`` and ``phi`` are in degrees.
    The returned absorption profile is unit-cell averaged and has units nm-1.
    """
    jnp, basis, fields, fmm, scattering = _imports()
    backend_options = options(user_options)

    lattice = np.asarray(size, dtype=float)
    if lattice.shape != (2, 2) or abs(np.linalg.det(lattice)) < 1e-12:
        raise ValueError("FMMAX requires two non-collinear 2D lattice vectors in nm.")
    if int(orders) != orders or orders < 1:
        raise ValueError(
            "The number of FMMAX Fourier orders must be a positive integer."
        )
    if len(widths) != len(layer_permittivities) or len(geom_list) != len(widths):
        raise ValueError(
            "Layer widths, permittivities, and geometry must have equal lengths."
        )
    if np.any(np.asarray(widths[1:-1], dtype=float) < 0):
        raise ValueError("Layer thicknesses must be non-negative.")

    primitive_lattice_vectors = basis.LatticeVectors(
        u=jnp.asarray(lattice[0]), v=jnp.asarray(lattice[1])
    )
    truncation = _enum_from_value(
        basis.Truncation, backend_options["truncation"], "truncation"
    )
    formulation = _enum_from_value(
        fmm.Formulation, backend_options["formulation"], "formulation"
    )
    expansion = basis.generate_expansion(
        primitive_lattice_vectors=primitive_lattice_vectors,
        approximate_num_terms=int(orders),
        truncation=truncation,
    )
    grid_shape = _grid_shape(lattice, expansion, backend_options, basis)
    permittivity_grids = _permittivity_grids(
        geom_list=geom_list,
        layer_permittivities=layer_permittivities,
        shape_permittivities=shape_permittivities,
        shape_names=shape_names,
        lattice=lattice,
        shape=grid_shape,
    )

    theta_rad = np.deg2rad(theta)
    phi_rad = np.deg2rad(phi)
    if not 0 <= theta_rad < np.pi / 2:
        raise ValueError(
            "FMMAX supports incidence angles from 0 (inclusive) to pi/2 (exclusive)."
        )
    if abs(np.imag(layer_permittivities[0])) > 1e-12:
        raise NotImplementedError(
            "FMMAX currently requires a non-absorbing incidence medium."
        )

    in_plane_wavevector = basis.plane_wave_in_plane_wavevector(
        wavelength=jnp.asarray(wavelength),
        polar_angle=jnp.asarray(theta_rad),
        azimuthal_angle=jnp.asarray(phi_rad),
        permittivity=jnp.asarray(layer_permittivities[0]),
    )
    layer_solve_results = [
        fmm.eigensolve_isotropic_media(
            wavelength=jnp.asarray(wavelength),
            in_plane_wavevector=in_plane_wavevector,
            primitive_lattice_vectors=primitive_lattice_vectors,
            permittivity=jnp.asarray(permittivity),
            expansion=expansion,
            formulation=formulation,
        )
        for permittivity in permittivity_grids
    ]
    thicknesses = (
        [jnp.asarray(0.0)]
        + [jnp.asarray(float(x)) for x in widths[1:-1]]
        + [jnp.asarray(0.0)]
    )
    s_matrices_interior = scattering.stack_s_matrices_interior(
        layer_solve_results=layer_solve_results,
        layer_thicknesses=thicknesses,
        force_x64_solve=bool(backend_options["force_x64_solve"]),
    )

    incident_amplitude, weights = _incident_amplitudes(
        layer_solve_results[0], theta_rad, phi_rad, pol, fields, jnp
    )
    zero_amplitude = jnp.zeros_like(incident_amplitude)
    incident_flux = np.sum(
        np.asarray(
            fields.amplitude_poynting_flux(
                incident_amplitude, zero_amplitude, layer_solve_results[0]
            )[0]
        ),
        axis=0,
    )
    if np.any(incident_flux <= 0):
        raise ValueError("The selected excitation has non-positive incident power.")

    amplitudes_interior = fields.stack_amplitudes_interior(
        s_matrices_interior=s_matrices_interior,
        forward_amplitude_0_start=incident_amplitude,
        backward_amplitude_N_end=zero_amplitude,
    )
    reflected_amplitude = s_matrices_interior[0][1].s21 @ incident_amplitude
    transmitted_amplitude = s_matrices_interior[-1][0].s11 @ incident_amplitude
    reflected_order = _weighted_normalized(
        _order_flux(
            reflected_amplitude, layer_solve_results[0], "backward", fields, jnp
        ),
        incident_flux,
        weights,
    )
    transmitted_order = _weighted_normalized(
        _order_flux(
            transmitted_amplitude, layer_solve_results[-1], "forward", fields, jnp
        ),
        incident_flux,
        weights,
    )

    absorption = []
    for layer_index in range(1, len(layer_solve_results) - 1):
        solve_result = layer_solve_results[layer_index]
        forward_start, backward_end = amplitudes_interior[layer_index]
        thickness = thicknesses[layer_index]
        forward_top, backward_top = fields.colocate_amplitudes(
            forward_start, backward_end, jnp.asarray(0.0), solve_result, thickness
        )
        forward_bottom, backward_bottom = fields.colocate_amplitudes(
            forward_start, backward_end, thickness, solve_result, thickness
        )
        top_flux = fields.amplitude_poynting_flux(
            forward_top, backward_top, solve_result
        )
        bottom_flux = fields.amplitude_poynting_flux(
            forward_bottom, backward_bottom, solve_result
        )
        absorbed = np.sum(
            np.asarray(top_flux[0] + top_flux[1] - bottom_flux[0] - bottom_flux[1]),
            axis=0,
        )
        absorption.append(np.sum(absorbed / incident_flux * weights))

    result = {
        "R": float(np.sum(reflected_order)),
        "T": float(np.sum(transmitted_order)),
        "A_per_layer": np.asarray(absorption, dtype=float),
        "R_per_order": np.asarray(reflected_order, dtype=float),
        "T_per_order": np.asarray(transmitted_order, dtype=float),
        "basis_set": np.asarray(expansion.basis_coefficients),
        "reciprocal": np.asarray(
            [
                primitive_lattice_vectors.reciprocal.u,
                primitive_lattice_vectors.reciprocal.v,
            ]
        ),
        "grid_shape": grid_shape,
        "version": version(),
    }

    transverse_wavevectors = np.asarray(
        basis.transverse_wavevectors(
            in_plane_wavevector, primitive_lattice_vectors, expansion
        )
    )
    result["transverse_wavevectors"] = transverse_wavevectors

    if profile_depths is not None:
        profile_depths = np.asarray(profile_depths, dtype=float)
        total_thickness = float(np.sum(widths[1:-1]))
        if np.any(profile_depths < 0) or np.any(profile_depths >= total_thickness):
            raise ValueError(
                "FMMAX profile depths must lie within the finite stack: "
                "0 <= z < total thickness."
            )
        layer_ends = np.cumsum(np.asarray(widths[1:-1], dtype=float))
        layer_starts = np.insert(layer_ends[:-1], 0, 0.0)
        profile = np.zeros(profile_depths.shape, dtype=float)
        angular_frequency = 2 * np.pi / wavelength
        for profile_index, depth in np.ndenumerate(profile_depths):
            finite_index = int(np.searchsorted(layer_ends, depth, side="right"))
            layer_index = finite_index + 1
            depth_in_layer = depth - layer_starts[finite_index]
            solve_result = layer_solve_results[layer_index]
            forward_amplitude, backward_amplitude = fields.colocate_amplitudes(
                *amplitudes_interior[layer_index],
                jnp.asarray(depth_in_layer),
                solve_result,
                thicknesses[layer_index],
            )
            electric_field, magnetic_field = fields.fields_from_wave_amplitudes(
                forward_amplitude, backward_amplitude, solve_result
            )
            electric_grid, _, _ = fields.fields_on_grid(
                electric_field=electric_field,
                magnetic_field=magnetic_field,
                layer_solve_result=solve_result,
                shape=grid_shape,
            )
            electric_intensity = sum(np.abs(np.asarray(x)) ** 2 for x in electric_grid)
            epsilon_imag = np.imag(permittivity_grids[layer_index])[..., np.newaxis]
            absorbed_density = (
                0.5
                * angular_frequency
                * np.mean(epsilon_imag * electric_intensity, axis=(0, 1))
            )
            profile[profile_index] = np.sum(absorbed_density / incident_flux * weights)
        result["profile"] = profile

    return result

"""Coordinate system utilities using Astropy.

Provides conversions between GSE, GEO, and other coordinate systems
used in spacecraft tracking and solar wind analysis.
"""

import numpy as np
from astropy.coordinates import (
    CartesianRepresentation,
    GeocentricMeanEcliptic,
    ITRS,
    GCRS,
)
from astropy.time import Time
from astropy import units as u


def gse_to_geo(x_gse: np.ndarray, y_gse: np.ndarray, z_gse: np.ndarray,
               times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert GSE (Geocentric Solar Ecliptic) to GEO (Geographic) coordinates.

    Args:
        x_gse, y_gse, z_gse: Position arrays in km (GSE frame)
        times: Array of datetime64 timestamps

    Returns:
        Tuple of (x_geo, y_geo, z_geo) in km
    """
    astro_times = Time(times)
    cart = CartesianRepresentation(x_gse * u.km, y_gse * u.km, z_gse * u.km)
    gse_coords = GeocentricMeanEcliptic(cart, obstime=astro_times)
    geo_coords = gse_coords.transform_to(ITRS(obstime=astro_times))

    return (
        geo_coords.cartesian.x.to(u.km).value,
        geo_coords.cartesian.y.to(u.km).value,
        geo_coords.cartesian.z.to(u.km).value,
    )


def compute_orbital_elements(
    x: np.ndarray, y: np.ndarray, z: np.ndarray,
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute classical orbital elements from Cartesian state vectors.

    Args:
        x, y, z: Position components in km
        vx, vy, vz: Velocity components in km/s

    Returns:
        Dict with orbital elements: semi_major_axis, eccentricity,
        inclination, raan, arg_periapsis, true_anomaly
    """
    MU_EARTH = 398600.4418  # km^3/s^2

    r_vec = np.column_stack([x, y, z])
    v_vec = np.column_stack([vx, vy, vz])

    r = np.linalg.norm(r_vec, axis=1)
    v = np.linalg.norm(v_vec, axis=1)

    # Specific angular momentum
    h_vec = np.cross(r_vec, v_vec)
    h = np.linalg.norm(h_vec, axis=1)

    # Node vector
    k_hat = np.array([0, 0, 1])
    n_vec = np.cross(k_hat, h_vec)
    n = np.linalg.norm(n_vec, axis=1)

    # Eccentricity vector
    e_vec = (np.cross(v_vec, h_vec) / MU_EARTH) - (r_vec / r[:, np.newaxis])
    eccentricity = np.linalg.norm(e_vec, axis=1)

    # Semi-major axis
    energy = v ** 2 / 2 - MU_EARTH / r
    semi_major_axis = -MU_EARTH / (2 * energy)

    # Inclination
    inclination = np.arccos(np.clip(h_vec[:, 2] / h, -1, 1))

    # RAAN (Right Ascension of Ascending Node)
    raan = np.arccos(np.clip(n_vec[:, 0] / np.maximum(n, 1e-10), -1, 1))
    raan[n_vec[:, 1] < 0] = 2 * np.pi - raan[n_vec[:, 1] < 0]

    # Argument of periapsis
    n_dot_e = np.sum(n_vec * e_vec, axis=1)
    arg_periapsis = np.arccos(np.clip(n_dot_e / np.maximum(n * eccentricity, 1e-10), -1, 1))
    arg_periapsis[e_vec[:, 2] < 0] = 2 * np.pi - arg_periapsis[e_vec[:, 2] < 0]

    # True anomaly
    e_dot_r = np.sum(e_vec * r_vec, axis=1)
    true_anomaly = np.arccos(np.clip(e_dot_r / np.maximum(eccentricity * r, 1e-10), -1, 1))
    rdot = np.sum(r_vec * v_vec, axis=1)
    true_anomaly[rdot < 0] = 2 * np.pi - true_anomaly[rdot < 0]

    return {
        "semi_major_axis_km": semi_major_axis,
        "eccentricity": eccentricity,
        "inclination_rad": inclination,
        "raan_rad": raan,
        "arg_periapsis_rad": arg_periapsis,
        "true_anomaly_rad": true_anomaly,
        "orbital_period_s": 2 * np.pi * np.sqrt(np.abs(semi_major_axis) ** 3 / MU_EARTH),
    }


def geocentric_distance(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute distance from Earth center in km."""
    return np.sqrt(x ** 2 + y ** 2 + z ** 2)


def geocentric_latitude(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Compute geocentric latitude in degrees."""
    r = geocentric_distance(x, y, z)
    return np.degrees(np.arcsin(np.clip(z / r, -1, 1)))

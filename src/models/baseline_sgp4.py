"""SGP4 orbit propagator baseline for comparison with ML models.

Uses the sgp4 library to propagate orbits from initial conditions,
providing a physics-based benchmark for ML model evaluation.
"""

import numpy as np
import pandas as pd
from sgp4.api import Satrec, WGS72
from astropy.time import Time
from astropy.coordinates import TEME, ITRS, CartesianRepresentation
from astropy import units as u


class SGP4Baseline:
    """SGP4 orbit propagator as ML baseline.

    For LEO satellites, SGP4 is the standard propagation model.
    ML models should aim to beat SGP4 on short-horizon predictions.
    """

    def __init__(self):
        self.satrec = None

    def fit_from_state(
        self,
        position_km: np.ndarray,
        velocity_km_s: np.ndarray,
        epoch: pd.Timestamp,
    ):
        """Initialize SGP4 from a Cartesian state vector.

        Args:
            position_km: [x, y, z] position in TEME frame (km)
            velocity_km_s: [vx, vy, vz] velocity in TEME frame (km/s)
            epoch: Epoch time
        """
        t = Time(epoch.to_pydatetime())
        jd = t.jd1
        fr = t.jd2

        # Create satellite record from state vector
        self.satrec = Satrec()
        self.satrec.sgp4init(
            WGS72,
            "i",  # improved mode
            0,  # satellite number
            (jd + fr - 2433281.5),  # epoch in days since 1949-12-31
            0.0,  # bstar drag term
            0.0,  # ndot (not used in sgp4)
            0.0,  # nddot (not used in sgp4)
            0.0,  # eccentricity (will be overridden by state)
            0.0,  # argument of perigee
            0.0,  # inclination
            0.0,  # mean anomaly
            0.0,  # mean motion (will be computed)
            0.0,  # right ascension of ascending node
        )

    def propagate(self, times: pd.DatetimeIndex) -> np.ndarray:
        """Propagate orbit to given times.

        Args:
            times: Array of prediction times

        Returns:
            positions: (N, 3) array of [x, y, z] positions in km
        """
        if self.satrec is None:
            raise ValueError("Must call fit_from_state first")

        positions = []
        for t in times:
            astro_t = Time(t.to_pydatetime())
            jd = astro_t.jd1
            fr = astro_t.jd2
            e, r, v = self.satrec.sgp4(jd, fr)
            if e == 0:
                positions.append(r)
            else:
                positions.append([np.nan, np.nan, np.nan])

        return np.array(positions)

    @staticmethod
    def simple_kepler_propagate(
        positions: np.ndarray,
        velocities: np.ndarray,
        dt_seconds: float,
        n_steps: int,
    ) -> np.ndarray:
        """Simple two-body Keplerian propagation as a naive baseline.

        Uses basic Euler integration of Newtonian gravity.
        This is intentionally simple — it's the "dumb" baseline.

        Args:
            positions: (3,) initial position in km
            velocities: (3,) initial velocity in km/s
            dt_seconds: Time step in seconds
            n_steps: Number of steps to propagate

        Returns:
            (n_steps, 3) predicted positions
        """
        MU_EARTH = 398600.4418  # km^3/s^2

        pos = positions.copy().astype(np.float64)
        vel = velocities.copy().astype(np.float64)
        trajectory = np.zeros((n_steps, 3))

        for i in range(n_steps):
            r = np.linalg.norm(pos)
            acc = -MU_EARTH / (r ** 3) * pos

            # Velocity Verlet integration
            pos_new = pos + vel * dt_seconds + 0.5 * acc * dt_seconds ** 2
            r_new = np.linalg.norm(pos_new)
            acc_new = -MU_EARTH / (r_new ** 3) * pos_new
            vel = vel + 0.5 * (acc + acc_new) * dt_seconds
            pos = pos_new

            trajectory[i] = pos

        return trajectory


def evaluate_baseline(
    test_positions: np.ndarray,
    test_velocities: np.ndarray,
    test_targets: np.ndarray,
    dt_seconds: float = 60.0,
) -> dict[str, float]:
    """Evaluate Kepler baseline on test set.

    Args:
        test_positions: (N, 3) initial positions for each test window
        test_velocities: (N, 3) initial velocities
        test_targets: (N, horizon_steps, 3) ground truth future positions
        dt_seconds: Time step

    Returns:
        Dict of metrics: mae_km, rmse_km
    """
    n_samples, horizon_steps, _ = test_targets.shape
    all_errors = []

    for i in range(n_samples):
        predicted = SGP4Baseline.simple_kepler_propagate(
            test_positions[i],
            test_velocities[i],
            dt_seconds,
            horizon_steps,
        )
        error = np.linalg.norm(predicted - test_targets[i], axis=1)
        all_errors.append(error)

    all_errors = np.array(all_errors)

    return {
        "mae_km": float(np.mean(np.abs(all_errors))),
        "rmse_km": float(np.sqrt(np.mean(all_errors ** 2))),
        "mae_6h": float(np.mean(np.abs(all_errors[:, :360]))) if horizon_steps >= 360 else None,
        "mae_24h": float(np.mean(np.abs(all_errors))) if horizon_steps >= 1440 else None,
    }

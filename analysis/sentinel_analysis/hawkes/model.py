"""Multivariate and Univariate Hawkes Process model implementation."""

import numpy as np

from sentinel_analysis.hawkes.kernels import Kernel


class HawkesProcess:
    """Hawkes process model supporting univariate and multivariate dimensions.

    Parameters:
        mu (Union[float, List[float], np.ndarray]): Baseline intensities per dimension (shape: (M,)).
        kernels (Union[Kernel, List[List[Kernel]]]): Memory kernels phi_{m, m'}(dt).
            If a single Kernel is provided, it is treated as a univariate model (M=1).
            If a 2D list of Kernels of shape (M, M) is provided, kernels[m][m'] specifies the kernel
            from triggering dimension m' to impacted dimension m.
    """

    def __init__(
        self,
        mu: float | list[float] | np.ndarray,
        kernels: Kernel | list[list[Kernel]],
    ):
        if isinstance(kernels, Kernel):
            # Univariate case
            self.dim = 1
            self.mu = np.array([float(mu)], dtype=np.float64)
            self.kernels = [[kernels]]
        else:
            # Multivariate case
            self.mu = np.atleast_1d(np.asarray(mu, dtype=np.float64))
            self.dim = len(self.mu)
            if len(kernels) != self.dim or any(len(row) != self.dim for row in kernels):
                raise ValueError(
                    f"Kernels matrix shape must be ({self.dim}, {self.dim}), "
                    f"got ({len(kernels)}, {len(kernels[0]) if kernels else 0})"
                )
            self.kernels = kernels

    def intensity(
        self,
        t: float,
        events: list[tuple[float, int]],
        dim: int | None = None,
    ) -> float | np.ndarray:
        """Compute conditional intensity lambda_m(t) given event history up to time t.

        Parameters:
            t (float): Evaluation timestamp.
            events (List[Tuple[float, int]]): History of past events [(timestamp, dim_index)].
            dim (Optional[int]): Specific dimension index (0 to M-1). If None, returns array for all M dimensions.

        Returns:
            Union[float, np.ndarray]: Intensity value for dimension `dim`, or array of intensities for all dimensions.
        """
        intensities = np.copy(self.mu)

        for t_i, m_prime in events:
            if t_i >= t:
                break
            dt = t - t_i
            for m in range(self.dim):
                intensities[m] += self.kernels[m][m_prime].eval(dt)

        if dim is not None:
            return float(intensities[dim])
        return intensities if self.dim > 1 else float(intensities[0])

    def log_likelihood(
        self,
        events: list[tuple[float, int]],
        T_end: float,
        T_start: float = 0.0,
        eps: float = 1e-12,
    ) -> float:
        """Compute log-likelihood log L(events | params) over time window [T_start, T_end].

        Formula:
            log L = sum_{m=1}^M sum_{i=1}^{N_m} log(max(lambda_m(t_{m, i}), eps)) - sum_{m=1}^M integral_{T_start}^{T_end} lambda_m(t) dt

        Parameters:
            events (List[Tuple[float, int]]): List of (timestamp, dimension_index) tuples, sorted by timestamp.
                If timestamps only are provided as float (for univariate), they will be converted to (t, 0).
            T_end (float): End timestamp of observation period.
            T_start (float): Start timestamp of observation period (default 0.0).
            eps (float): Small offset floor for numerical stability in log.

        Returns:
            float: Log-likelihood value.
        """
        # Standardize events format
        formatted_events: list[tuple[float, int]] = []
        for ev in events:
            if isinstance(ev, (int, float)):
                formatted_events.append((float(ev), 0))
            else:
                formatted_events.append((float(ev[0]), int(ev[1])))

        # Sort events by timestamp
        formatted_events.sort(key=lambda x: x[0])

        if T_end <= T_start:
            raise ValueError(
                f"T_end ({T_end}) must be strictly greater than T_start ({T_start})"
            )

        # 1. Sum of log intensities at event times
        log_intensity_sum = 0.0
        past_events: list[tuple[float, int]] = []

        for t_i, m_i in formatted_events:
            if t_i < T_start or t_i > T_end:
                continue
            lambda_val = self.intensity(t_i, past_events, dim=m_i)
            log_intensity_sum += np.log(max(lambda_val, eps))
            past_events.append((t_i, m_i))

        # 2. Integrated intensity over [T_start, T_end]
        integrated_sum = np.sum(self.mu) * (T_end - T_start)

        for t_i, m_prime in formatted_events:
            if t_i < T_start or t_i >= T_end:
                continue
            dt_end = T_end - t_i
            for m in range(self.dim):
                integrated_sum += self.kernels[m][m_prime].integrate(dt_end)

        return float(log_intensity_sum - integrated_sum)

    def simulate(
        self,
        T_end: float,
        T_start: float = 0.0,
        rng: np.random.Generator | int | None = None,
        max_events: int = 1000000,
    ) -> list[tuple[float, int]]:
        """Simulate event stream using Ogata's modified Thinning Algorithm.

        Parameters:
            T_end (float): Simulation end time.
            T_start (float): Simulation start time (default 0.0).
            rng (Union[np.random.Generator, int, None]): Seed or explicit NumPy random generator.
            max_events (int): Safety cap on maximum generated events.

        Returns:
            List[Tuple[float, int]]: Generated event stream as list of (timestamp, dim_index) tuples.
        """
        if isinstance(rng, (int, np.integer)):
            gen = np.random.default_rng(int(rng))
        elif isinstance(rng, np.random.Generator):
            gen = rng
        else:
            gen = np.random.default_rng()

        if T_end <= T_start:
            raise ValueError(
                f"T_end ({T_end}) must be strictly greater than T_start ({T_start})"
            )

        events: list[tuple[float, int]] = []
        t = T_start

        while t < T_end and len(events) < max_events:
            # Current intensities at t+
            current_intensities = np.atleast_1d(self.intensity(t + 1e-12, events))
            lambda_bar = float(np.sum(current_intensities))

            if lambda_bar <= 0:
                break

            # Draw candidate wait time from exponential distribution
            s = gen.exponential(1.0 / lambda_bar)
            t = t + s

            if t >= T_end:
                break

            # Evaluate exact intensity at candidate time t
            candidate_intensities = np.atleast_1d(self.intensity(t, events))
            candidate_total = float(np.sum(candidate_intensities))

            u = gen.uniform(0.0, lambda_bar)

            if u <= candidate_total:
                # Accept event! Determine dimension index
                cum_sum = np.cumsum(candidate_intensities)
                accepted_dim = int(np.searchsorted(cum_sum, u))
                events.append((t, min(accepted_dim, self.dim - 1)))

        return events

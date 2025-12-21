from .noise import *
from backends import QubitTracking
import numpy as np


class ArtificialVarianceNoise(NoiseModel):
    m_base = 0.01
    m_sigma = 0.0

    @staticmethod
    def measure() -> float:
        """
        Log-normal distribution since it makes more sense here
        """
        if ArtificialVarianceNoise.m_sigma == 0.0:
            return ArtificialVarianceNoise.m_base

        mu_log = np.log(ArtificialVarianceNoise.m_base)
        value = np.random.lognormal(mean=mu_log,
                                     sigma=ArtificialVarianceNoise.m_sigma)

        return np.clip(value, 0.0, 1.0)

    @classmethod
    def get_noise(
        cls,
        qt: QubitTracking,
        backend,
        variance: str = "none"
    ) -> 'NoiseModel':
        if variance == "low":
            cls.m_sigma = 0.4
        elif variance == "mid":
            cls.m_sigma = 0.7
        elif variance == "high":
            cls.m_sigma = 1.0
        else:
            cls.m_sigma = 0.0

        return NoiseModel(
            sq=0.00025,
            tq=0.002,
            measure=cls.measure,  
            gate_times={
                "SQ": 50e-9,
                "TQ": 70e-9,
                "M": 70e-9,
                "R": 1.2942222222222222e-6
            },
            qt=qt,
            backend=backend
        )

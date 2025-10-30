from .noise import *
from backends import FakeIBMFlamingo, QubitTracking


class FlamingoNoise(NoiseModel):

    @staticmethod
    def get_noise(
        qt: QubitTracking,
        backend: FakeIBMFlamingo,
        m_error_multiplier = 1,
        m_time_multiplier = 1
    ) -> 'NoiseModel':
        m_error_multiplier = float(m_error_multiplier)
        m_time_multiplier = float(m_time_multiplier)
        return NoiseModel(
            sq=0.00025,
            tq=0.002 / 10, # TODO
            measure=0.01 * m_error_multiplier,
            remote=0.03,
            gate_times={
                "SQ": 50 * 1e-9,
                "TQ": 70 * 1e-9,
                "M": 1000 * 1e-9 * m_time_multiplier,
                "REMOTE": (300 * 1e-9) / (2.2222222222222221e-10 * 1e9) * (2.2222222222222221e-10 * 1e9),
                "R": 1.2942222222222222e-06
            },
            qt=qt,
            backend=backend
        )

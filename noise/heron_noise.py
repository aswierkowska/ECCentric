from .noise import *
from backends import QubitTracking


class HeronNoise(NoiseModel):
    @staticmethod
    def get_noise(
        qt: QubitTracking,
        backend
    ) -> 'NoiseModel':
        return NoiseModel(
            sq=0.0002854,
            tq=0.002623,
            measure=0.0309,
            gate_times={
                "SQ": 32 * 1e-9,
                "TQ": 68 * 1e-9,
                "M": 1560 * 1e-9,
                "R": 1.2942222222222222e-06
            },
            qt=qt,
            backend=backend
        )


'''def get_noise(
        qt: QubitTracking,
        backend
    ) -> 'NoiseModel':
        return NoiseModel(
            sq=0.00025,
            tq=0.002,
            measure=0.01,
            gate_times={
                "SQ": 50 * 1e-9,
                "TQ": 70 * 1e-9,
                "M": 70 * 1e-9,
                "R": 1.2942222222222222e-06
            },
            qt=qt,
            backend=backend
        )'''
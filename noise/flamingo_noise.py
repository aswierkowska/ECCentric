from .noise import *
from backends import FakeIBMFlamingo, QubitTracking


class FlamingoNoise(NoiseModel):

    @staticmethod
    def get_noise(
        qt: QubitTracking,
        backend: FakeIBMFlamingo,
        scaling: int = 1
    ) -> 'NoiseModel':
        return NoiseModel(
            sq=0.00025 / scaling,
            tq=0.002 / scaling,
            measure=0.01 / scaling,
            remote=0.03 / scaling,
            gate_times={
                "SQ": 50 * 1e-9,
                "TQ": 70 * 1e-9,
                "M": 70 * 1e-9,
                "REMOTE": round((300 * 1e-9) / (2.2222222222222221e-10 * 1e9)) * (2.2222222222222221e-10 * 1e9),
                "R": 1.2942222222222222e-06
            },
            qt=qt,
            backend=backend
        )
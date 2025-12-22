import logging
import stim
import numpy as np

from .bp_osd import bposd_chk, bposd_batch
from .mwpm import mwpm
from .chromobius import chromobius


def decode(
        code_name: str, circuit: stim.Circuit, num_shots: int, decoder: str, backend_name: str, error_type: str
) -> float:
    approximate_disjoint_errors = False
    if (backend_name.split("_")[0] == "variance" or backend_name.split("_")[0] == "real") and (error_type == "variance" or error_type.split("_")[0] == "real"):
        approximate_disjoint_errors = True

    if decoder == "mwpm":
        try:
            return mwpm(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"MWPM failed: {e}")
    elif decoder == "chromobius":
        try:
            return chromobius(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"Chromobius failed: {e}")
    elif decoder == "bposd":
        try:
            if code_name == "steane" or code_name == "bacon":
                return bposd_chk(circuit, num_shots, approximate_disjoint_errors)
            else:
                return bposd_batch(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"BP-OSD failed: {e}")
            return None
    elif decoder == "bposd_chk":
        try:
            return bposd_chk(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"BP-OSD CHK failed: {e}")
            return None
    elif decoder == "bposd_batch":
        try:
            return bposd_batch(circuit, num_shots, approximate_disjoint_errors)
        except Exception as e:
            logging.error(f"BP-OSD BATCH failed: {e}")
            return None
    else:
        raise NotImplementedError

def raw_error_rate(circuit: stim.Circuit, num_shots: int):
    sampler = circuit.compile_detector_sampler()
    _, observable_flips = sampler.sample(
        num_shots, separate_observables=True
    )

    errors = np.any(observable_flips, axis=1)
    num_errors = np.count_nonzero(errors)

    return num_errors
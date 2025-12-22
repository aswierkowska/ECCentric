import stim
import numpy as np
import chromobius

def chromobius(circuit: stim.Circuit, num_shots: int, approximate_disjoint_errors: bool):
        dets, actual_obs_flips = circuit.compile_detector_sampler().sample(
            shots=num_shots,
            separate_observables=True,
            bit_packed=True,
        )
        decoder = chromobius.compile_decoder_for_dem(
            circuit.detector_error_model(), 
            approximate_disjoint_errors=approximate_disjoint_errors
        )
        predicted_obs_flips = decoder.predict_obs_flips_from_dets_bit_packed(dets)
        return np.count_nonzero(np.any(predicted_obs_flips != actual_obs_flips, axis=1)) / num_shots
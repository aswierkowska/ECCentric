import sys
import os
import stim

sys.path.append(os.path.join(os.getcwd(), "external/qiskit_qec/src"))

import yaml
import logging

from itertools import product
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import SamplerV2 as Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.compiler import transpile
from qiskit_qec.utils import get_stim_circuits
from backends import get_backend, QubitTracking
from codes import get_code, get_max_d, get_min_n
from noise import get_noise_model
from decoders import decode, raw_error_rate
from transpilers import run_transpiler, translate
from utils import save_experiment_metadata, save_results_to_csv, setup_experiment_logging
import stim


def load_API_key():
 
    QiskitRuntimeService.save_account(
    token="<token>", # Use the 44-character API_KEY you created and saved from the IBM Quantum Platform Home dashboard
    instance="open-instance", # Optional
    overwrite=True
    )

def run_on_real_device(circuit):

    service = QiskitRuntimeService()
    #backend = service.least_busy(simulator=False, operational=True)
    backend = service.backend("ibm_torino")
 
    # Convert to an ISA circuit and layout-mapped observables.
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(circuit)


    #estimator = Estimator(mode=backend)
    #estimator.options.resilience_level = 1
    #estimator.options.default_shots = 5000

    
    sampler = Sampler(mode=backend)

    observables_labels = ["IIZIIIIIIZIZIZZIZIZIIIIIIZ"]
    observables = [SparsePauliOp(label) for label in observables_labels]

    mapped_observables = [
    observable.apply_layout(isa_circuit.layout) for observable in observables
    ]
 
    #job = estimator.run([(isa_circuit, mapped_observables)])
    job = sampler.run([(isa_circuit, None)], shots=5000)
 
    # Use the job ID to retrieve your job data later
    print(f">>> Job ID: {job.job_id()}")
    pass

def run_experiment(
    experiment_name,
    backend_name,
    backend_size,
    code_name,
    decoder,
    d,
    cycles,
    num_samples,
    error_type,
    error_prob,
    lock,
    layout_method=None,
    routing_method=None,
    translating_method=None,
):
    try:
        print(f"Starting experiment")
        backend = get_backend(backend_name, backend_size)
        print("Got backend")
        if d == None:
            if backend_name == "real_flamingo_1_qpu":
                d = get_max_d(code_name, 133)
            elif backend_name == "real_loon_1_qpu":
                d = get_max_d(code_name, 120)
            else:
                d = get_max_d(code_name, backend.coupling_map.size())
            print(f"Max distance for {code_name} on backend {backend_name} is {d}")
            if d < 3:
                logging.info(
                    f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}: Execution not possible"
                )
                return
        
        #if cycles is not None and cycles <= 1:
        #    logging.info(
        #        f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}: Execution not possible, cycles must be greater than 1"
        #    )
        #    return
        
        if cycles is None:
            cycles = d
        
              
        print("Got distance")
        code = get_code(code_name, d, cycles)
        print(f"Got code")
        detectors, logicals = code.stim_detectors()
        print("Before translating")

        if translating_method:
            code.qc = translate(code.qc, translating_method)
        code.qc = run_transpiler(code.qc, backend, layout_method, routing_method)

        print(code.qc)

        #from qiskit.primitives import StatevectorSampler
        #sampler = StatevectorSampler()
        #result = sampler.run([code.qc], shots=1024).result()
        #print(result[0].data.meas.get_counts())

        load_API_key()
        run_on_real_device(code.qc)

        qt = QubitTracking(backend, code.qc)
        stim_circuit = get_stim_circuits(
            code.qc, detectors=detectors, logicals=logicals
        )[0][0]
        noise_model = get_noise_model(error_type, qt, error_prob, backend)
        stim_circuit = noise_model.noisy_circuit(stim_circuit)
        #logical_error_rate = decode(code_name, stim_circuit, num_samples, decoder, backend_name, error_type)
        logical_error_rate = raw_error_rate(stim_circuit, num_samples) / num_samples

        result_data = {
            "backend": backend_name,
            "backend_size": backend_size,
            "code": code_name,
            "decoder": decoder,
            "distance": d,
            "cycles": cycles if cycles else d,
            "num_samples": num_samples,
            "error_type": error_type,
            "error_probability": error_prob,
            "logical_error_rate": f"{logical_error_rate:.3f}",
            "layout_method": layout_method if layout_method else "N/A",
            "routing_method": routing_method if routing_method else "N/A",
            "translating_method": translating_method if translating_method else "N/A"
        }

        with lock:
            save_results_to_csv(result_data, experiment_name)


        if backend_size:
            logging.info(
                f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name} {backend_size}, error type {error_type}, decoder {decoder}: {logical_error_rate:.6f}"
            )
        else:
            logging.info(
                f"{experiment_name} | Logical error rate for {code_name} with distance {d}, backend {backend_name}, error type {error_type}, decoder {decoder}: {logical_error_rate:.6f}"
            )

    except Exception as e:
            logging.error(
                f"{experiment_name} | Failed to run experiment for {code_name}, distance {d}, backend {backend_name}, error type {error_type}: {e}"
            )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Remember to add YAML file!")
        sys.exit(1)
    
    conf_file = sys.argv[1]

    with open(conf_file, "r") as f:
        config = yaml.safe_load(f)

    for experiment in config["experiments"]:
        experiment_name = experiment["name"]
        num_samples = experiment["num_samples"]
        backends = experiment["backends"]
        codes = experiment["codes"]
        decoders = experiment["decoders"]
        error_types = experiment["error_types"]
        error_probabilities = experiment.get("error_probabilities", [None])
        cycles = experiment.get("cycles", None)
        layout_methods = experiment.get("layout_methods", [None])
        routing_methods = experiment.get("routing_methods", [None])
        translating_methods = experiment.get("translating_methods", [None])

        setup_experiment_logging(experiment_name)
        save_experiment_metadata(experiment, experiment_name)
        manager = Manager()
        lock = manager.Lock()
        # TODO: better handling case if distances and backends_sizes are both set

        with ProcessPoolExecutor() as executor:
            if "backends_sizes" in experiment and "distances" in experiment:
                raise ValueError("Cannot set both backends_sizes and distances in the same experiment")
            if "distances" in experiment:
                distances = experiment["distances"]
                parameter_combinations = product(backends, codes, decoders, error_types, error_probabilities, distances, layout_methods, routing_methods, translating_methods)
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        get_min_n(code_name, d),
                        code_name,
                        decoder,
                        d,
                        cycles,
                        num_samples,
                        error_type,
                        error_prob,
                        lock,
                        layout_method,
                        routing_method,
                        translating_method
                    )
                    for backend, code_name, decoder, error_type, error_prob, d, layout_method, routing_method, translating_method in parameter_combinations
                ]
            elif "backends_sizes" in experiment:
                backends_sizes = experiment["backends_sizes"]
                parameter_combinations = product(
                    backends, backends_sizes, codes, decoders, error_types, error_probabilities, layout_methods, routing_methods, translating_methods
                )
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        backends_sizes,
                        code_name,
                        decoder,
                        None,
                        cycles,
                        num_samples,
                        error_type,
                        error_prob,
                        lock,
                        layout_method,
                        routing_method,
                        translating_method,
                    )
                    for backend, backends_sizes, code_name, decoder, error_type, error_prob, layout_method, routing_method, translating_method in parameter_combinations
                ]
            else:
                parameter_combinations = product(backends, codes, decoders, error_types, error_probabilities, layout_methods, routing_methods, translating_methods)
                futures = [
                    executor.submit(
                        run_experiment,
                        experiment_name,
                        backend,
                        None,
                        code_name,
                        decoder,
                        None,
                        cycles,
                        num_samples,
                        error_type,
                        error_prob,
                        lock,
                        layout_method,
                        routing_method,
                        translating_method,
                    )
                    for backend, code_name, decoder, error_type, error_prob, layout_method, routing_method, translating_method in parameter_combinations
                ]
            for future in futures:
                future.result()

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import Estimator, Sampler
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as QiskitNoiseModel
from qiskit.result import Result
from qiskit.circuit.library import XGate, SXGate, RZGate, CXGate, IGate
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime

from noise import get_noise_model
from noise.heron_noise import HeronNoise
from noise.artificial_noise import ArtificialNoise
from backends import QubitTracking


# ========================= IBM CLOUD UTILITIES =========================

def get_available_ibm_backends(
    min_qubits: int = 2,
    operational_only: bool = True
) -> List[Dict[str, Any]]:
    """Get list of available IBM backends with their properties."""
    try:
        service = QiskitRuntimeService()
        backends = service.backends()
        
        available = []
        for backend in backends:
            if backend.num_qubits >= min_qubits:
                status = backend.status()
                if not operational_only or status.operational:
                    backend_info = {
                        'name': backend.name,
                        'num_qubits': backend.num_qubits,
                        'operational': status.operational,
                        'pending_jobs': status.pending_jobs,
                        'status_msg': status.status_msg
                    }
                    try:
                        backend_info['queue_length'] = getattr(status, 'queue_length', 'N/A')
                    except:
                        backend_info['queue_length'] = 'N/A'
                    available.append(backend_info)
        
        # Sort by queue length (if available) and number of qubits
        available.sort(key=lambda x: (
            x['pending_jobs'] if x['pending_jobs'] is not None else 999,
            -x['num_qubits']
        ))
        
        return available
    
    except Exception as e:
        print(f"Error fetching IBM backends: {e}")
        return []


def select_best_ibm_backend(
    min_qubits: int = 2,
    preferred_backends: List[str] = None
) -> Optional[str]:
    """Select the best available IBM backend based on queue and preferences."""
    
    if preferred_backends is None:
        preferred_backends = ["ibm_torino", "ibm_heron", "ibm_flamingo"]
    
    available = get_available_ibm_backends(min_qubits=min_qubits)
    
    if not available:
        print("No available IBM backends found")
        return None
    
    print("\nAvailable IBM backends:")
    for backend in available:
        print(f"  {backend['name']}: {backend['num_qubits']} qubits, "
              f"queue: {backend['pending_jobs']}, operational: {backend['operational']}")
    
    # First try preferred backends
    for preferred in preferred_backends:
        for backend in available:
            if backend['name'] == preferred and backend['operational']:
                print(f"\nSelected preferred backend: {preferred}")
                return preferred
    
    # Fall back to best available (lowest queue, operational)
    for backend in available:
        if backend['operational']:
            print(f"\nSelected best available backend: {backend['name']}")
            return backend['name']
    
    print("No operational backends found")
    return None


def estimate_job_cost(
    num_circuits: int,
    shots_per_circuit: int,
    backend_name: str = "ibm_torino"
) -> Dict[str, Any]:
    """Estimate the cost/time for an IBM cloud job."""
    
    total_shots = num_circuits * shots_per_circuit
    
    # Rough estimates - these would need to be updated with current pricing
    cost_estimates = {
        'total_shots': total_shots,
        'estimated_queue_time_minutes': 'depends on queue',
        'estimated_execution_time_minutes': total_shots / 1000,  # Very rough estimate
        'recommendation': []
    }
    
    if total_shots > 100000:
        cost_estimates['recommendation'].append("Consider reducing shots or splitting into multiple jobs")
    
    if shots_per_circuit < 512:
        cost_estimates['recommendation'].append("Very low shots may give poor statistics")
    elif shots_per_circuit > 2048:
        cost_estimates['recommendation'].append("High shots - ensure necessary for your analysis")
    
    return cost_estimates


# ========================= IBM CLOUD FUNCTIONS =========================

def submit_circuits_to_ibm_cloud(
    circuits: List[QuantumCircuit],
    backend_name: str = "ibm_torino",
    shots_per_circuit: int = 1024,
    optimization_level: int = 1,
    job_name: Optional[str] = None,
    save_job_info: bool = True,
    max_shots_per_job: int = 100000  # IBM cloud limit
) -> Dict[str, Any]:
    """
    Submit a batch of circuits to IBM cloud as a single job.
    
    Parameters:
    -----------
    circuits : List[QuantumCircuit]
        List of circuits to execute
    backend_name : str
        IBM backend name (e.g., "ibm_torino", "ibm_heron")
    shots_per_circuit : int
        Number of shots per circuit
    optimization_level : int
        Transpilation optimization level (0-3)
    job_name : str, optional
        Custom job name for tracking
    save_job_info : bool
        Whether to save job information to file
    max_shots_per_job : int
        Maximum total shots per IBM job (to respect limits)
        
    Returns:
    --------
    Dict with job information including job_id for later retrieval
    """
    
    # Initialize runtime service
    try:
        service = QiskitRuntimeService()
        backend = service.backend(backend_name)
    except Exception as e:
        print(f"Error connecting to IBM cloud: {e}")
        raise
    
    print(f"Using IBM backend: {backend_name}")
    print(f"Backend status: {backend.status()}")
    
    # Calculate optimal batch size to respect shot limits
    total_requested_shots = len(circuits) * shots_per_circuit
    if total_requested_shots > max_shots_per_job:
        # Reduce shots per circuit to fit within limits
        adjusted_shots = max_shots_per_job // len(circuits)
        print(f"WARNING: Reducing shots from {shots_per_circuit} to {adjusted_shots} to fit IBM limits")
        shots_per_circuit = adjusted_shots
    
    print(f"Submitting {len(circuits)} circuits with {shots_per_circuit} shots each")
    print(f"Total shots for this job: {len(circuits) * shots_per_circuit}")
    
    # Ensure all circuits have measurements
    measured_circuits = []
    for i, circuit in enumerate(circuits):
        if not any(instr.operation.name == 'measure' for instr in circuit.data):
            circuit_with_measurements = circuit.copy()
            if not circuit_with_measurements.cregs:
                circuit_with_measurements.add_register('c', circuit.num_qubits)
            circuit_with_measurements.measure_all()
            circuit = circuit_with_measurements
        measured_circuits.append(circuit)
    
    # Transpile all circuits for the target backend
    print("Transpiling circuits for IBM backend...")
    transpiled_circuits = transpile(
        measured_circuits,
        backend=backend,
        optimization_level=optimization_level
    )
    
    # Create job name with timestamp if not provided
    if job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job_name = f"noise_comparison_{len(circuits)}circuits_{timestamp}"
    
    # Submit job using current IBM Runtime API
    print(f"Submitting job '{job_name}' to {backend_name}...")
    
    try:
        # Use the current Sampler primitive
        sampler = Sampler(backend)
        job = sampler.run(transpiled_circuits, shots=shots_per_circuit)
    except Exception as e:
        raise RuntimeError(f"Could not submit job to {backend_name}. Error: {e}")
    
    job_info = {
        'job_id': job.job_id(),
        'backend_name': backend_name,
        'job_name': job_name,
        'num_circuits': len(circuits),
        'shots_per_circuit': shots_per_circuit,
        'total_shots': len(circuits) * shots_per_circuit,
        'submission_time': datetime.now().isoformat(),
        'optimization_level': optimization_level,
        'circuit_depths': [c.depth() for c in transpiled_circuits],
        'circuit_sizes': [c.size() for c in transpiled_circuits],
        'transpiled_circuits': transpiled_circuits  # Store for later analysis
    }
    
    print(f"Job submitted successfully!")
    print(f"Job ID: {job.job_id()}")
    print(f"Job status: {job.status()}")
    
    # Save job information to file for later retrieval
    if save_job_info:
        job_file = f"job_info_{job.job_id()}.json"
        # Convert circuits to serializable format
        serializable_job_info = job_info.copy()
        del serializable_job_info['transpiled_circuits']  # Can't serialize circuits directly
        try:
            serializable_job_info['circuit_qasm'] = [c.qasm() for c in transpiled_circuits]
        except AttributeError:
            # Fallback for older qiskit versions
            serializable_job_info['circuit_qasm'] = [str(c) for c in transpiled_circuits]
        
        with open(job_file, 'w') as f:
            json.dump(serializable_job_info, f, indent=2)
        print(f"Job information saved to: {job_file}")
    
    return job_info


def get_ibm_job_results(
    job_id: str,
    wait_for_completion: bool = True,
    timeout: int = 3600
) -> Dict[str, Any]:
    """
    Retrieve results from an IBM cloud job.
    
    Parameters:
    -----------
    job_id : str
        IBM job ID
    wait_for_completion : bool
        Whether to wait for job completion
    timeout : int
        Maximum time to wait for completion (seconds)
        
    Returns:
    --------
    Dict with job results and metadata
    """
    
    # Initialize runtime service
    service = QiskitRuntimeService()
    
    try:
        job = service.job(job_id)
        print(f"Retrieved job {job_id}")
        
        # Handle different ways job status might be returned
        job_status = job.status()
        if hasattr(job_status, 'name'):
            status_name = job_status.name
        else:
            status_name = str(job_status)  # Handle case where status is already a string
        
        print(f"Job status: {status_name}")
        
        if wait_for_completion:
            print(f"Waiting for job completion (timeout: {timeout}s)...")
            result = job.result(timeout=timeout)
        else:
            if status_name not in ['DONE', 'CANCELLED', 'ERROR']:
                print("Job not completed yet. Set wait_for_completion=True to wait.")
                return {'status': status_name, 'job_id': job_id}
            result = job.result()
        
        # Extract counts for each circuit
        counts_list = []
        if hasattr(result, '__len__'):  # Multiple circuits
            for i in range(len(result)):
                counts_list.append(result[i].data.meas.get_counts())
        else:  # Single circuit
            counts_list.append(result.data.meas.get_counts())
        
        # Get job metadata
        metadata = {
            'job_id': job_id,
            'status': status_name,
            'backend_name': job.backend().name,
            'creation_time': job.creation_date.isoformat() if job.creation_date else None,
            'completion_time': datetime.now().isoformat(),
            'num_circuits': len(counts_list),
        }
        
        # Try to get additional metadata
        try:
            metadata.update({
                'queue_position': getattr(job, 'queue_position', None),
                'execution_time': getattr(job, 'time_taken', None),
            })
        except:
            pass
        
        return {
            'metadata': metadata,
            'counts': counts_list,
            'raw_result': result
        }
        
    except Exception as e:
        print(f"Error retrieving job results: {e}")
        return {'error': str(e), 'job_id': job_id}


def load_job_info_from_file(job_id: str) -> Dict[str, Any]:
    """Load job information from saved JSON file."""
    job_file = f"job_info_{job_id}.json"
    if os.path.exists(job_file):
        with open(job_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Job info file not found: {job_file}")
        return {}


def compare_circuit_execution(
    circuit: QuantumCircuit,
    shots: int = 1024,
    noise_type: str = "heron", 
    noise_param: float = 0.001,
    custom_backend=None,
    custom_noise_model=None
) -> Dict[str, Any]:
    """
    Compare circuit execution across three different backends:
    1. Noiseless simulator
    2. FakeTorino backend (IBM's fake backend with built-in noise)
    3. Custom backend with provided noise model
    
    IMPORTANT: All backends execute the EXACT same transpiled circuit for fair comparison.
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        The quantum circuit to execute
    shots : int
        Number of shots for execution (default: 1024)
    noise_type : str
        Type of noise model to use ('heron', 'flamingo', 'infleqtion', etc.)
    noise_param : float  
        Noise parameter (if applicable for the noise type)
    custom_backend : BackendV2, optional
        Custom backend to use (if None, will use a default based on noise_type)
    custom_noise_model : NoiseModel, optional
        Custom noise model to override the default
        
    Returns:
    --------
    Dict containing results from all three backends with execution statistics
    """
    
    results = {}
    
    # Ensure circuit has measurements
    if not any(instr.operation.name == 'measure' for instr in circuit.data):
        # Add measurements to all qubits
        circuit_with_measurements = circuit.copy()
        circuit_with_measurements.add_register(circuit_with_measurements.cregs[0] if circuit_with_measurements.cregs else circuit_with_measurements.add_register('c', circuit.num_qubits))
        circuit_with_measurements.measure_all()
        circuit = circuit_with_measurements
    
    # CRITICAL: Transpile once to get identical circuit for all backends
    fake_torino = FakeTorino()
    
    # For Heron devices, we should use CZ instead of CX
    # Transpile to a common basis set that both backends can handle
    common_transpiled = transpile(
        circuit, 
        fake_torino,
        basis_gates=['sx', 'x', 'rz', 'cz', 'id'],  # Use CZ for Heron compatibility
        optimization_level=1
    )
    
    print(f"\nTranspiled circuit depth: {common_transpiled.depth()}")
    print(f"Transpiled circuit gates: {common_transpiled.count_ops()}")
    
    # 1. Noiseless Simulator
    print("Running on noiseless simulator...")
    noiseless_simulator = AerSimulator()
    # Use the same transpiled circuit for fair comparison
    noiseless_job = noiseless_simulator.run(common_transpiled, shots=shots)
    noiseless_result = noiseless_job.result()
    
    results['noiseless'] = {
        'backend': 'AerSimulator (noiseless)',
        'counts': noiseless_result.get_counts(),
        'fidelity': 1.0,  # Noiseless is always perfect fidelity
        'execution_time': getattr(noiseless_result, 'time_taken', 'N/A')
    }
    
    # 2. FakeTorino Backend
    print("Running on FakeTorino backend...")
    
    # Create simulator with FakeTorino's noise model
    torino_noise_model = QiskitNoiseModel.from_backend(fake_torino)
    
    # Extract noise parameters for comparison
    torino_noise_details = _extract_torino_noise_parameters(torino_noise_model, fake_torino)
    
    torino_simulator = AerSimulator(noise_model=torino_noise_model)
    
    # Use the same transpiled circuit for fair comparison
    torino_job = torino_simulator.run(common_transpiled, shots=shots)
    torino_result = torino_job.result()
    
    # Calculate fidelity relative to noiseless
    torino_fidelity = _calculate_fidelity(
        results['noiseless']['counts'], 
        torino_result.get_counts()
    )
    
    results['fake_torino'] = {
        'backend': 'FakeTorino with noise',
        'counts': torino_result.get_counts(),
        'fidelity': torino_fidelity,
        'execution_time': getattr(torino_result, 'time_taken', 'N/A'),
        'coupling_map': fake_torino.coupling_map,
        'num_qubits': fake_torino.num_qubits,
        'torino_noise_details': torino_noise_details
    }
    
    # 3. Custom Backend with Custom Noise Model
    print(f"Running on custom backend with {noise_type} noise...")
    
    # Set up backend first
    if custom_backend is None:
        custom_backend = fake_torino  # Use Torino architecture as base
    
    # Set up QubitTracking with proper parameters using the common transpiled circuit
    qt = QubitTracking(custom_backend, common_transpiled)
    
    # Get the appropriate noise model
    if noise_type == "heron":
        noise_model = HeronNoise.get_noise(qt, custom_backend)
    else:
        # Use the generic noise model getter
        noise_model = get_noise_model(noise_type, qt, noise_param, custom_backend)
    
    if custom_noise_model is not None:
        noise_model = custom_noise_model
    
    # Create a Qiskit noise model approximation that uses CZ instead of CX for Heron
    qiskit_custom_noise = _create_qiskit_noise_approximation(noise_model, custom_backend, use_cz=True)
    custom_simulator = AerSimulator(noise_model=qiskit_custom_noise)
    
    # Use the same transpiled circuit for fair comparison
    custom_job = custom_simulator.run(common_transpiled, shots=shots)
    custom_result = custom_job.result()
    
    # Calculate fidelity relative to noiseless
    custom_fidelity = _calculate_fidelity(
        results['noiseless']['counts'],
        custom_result.get_counts()
    )
    
    results['custom_noise'] = {
        'backend': f'Custom backend with {noise_type} noise',
        'counts': custom_result.get_counts(),
        'fidelity': custom_fidelity,
        'execution_time': getattr(custom_result, 'time_taken', 'N/A'),
        'noise_parameters': {
            'sq_error': getattr(noise_model, 'sq', 'N/A'),
            'tq_error': getattr(noise_model, 'tq', 'N/A'), 
            'measure_error': getattr(noise_model, 'measure', 'N/A'),
            'reset_error': getattr(noise_model, 'reset', 'N/A')
        }
    }
    
    # Summary comparison
    results['comparison'] = _generate_comparison_summary(results)
    
    return results


def compare_circuit_execution_with_ibm_cloud(
    circuit: QuantumCircuit,
    ibm_job_results: Dict[str, int],
    circuit_index: int = 0,
    shots: int = 1024,
    noise_types: List[str] = ["heron"],
    noise_param: float = 0.001,
    custom_backend=None,
    custom_noise_model=None,
    ibm_backend_name: str = "ibm_torino"
) -> Dict[str, Any]:
    """
    Compare circuit execution using IBM cloud results instead of FakeTorino.
    
    This function compares:
    1. Noiseless simulator
    2. IBM cloud real hardware results (provided)
    3. Multiple custom backends with different noise models
    
    Parameters:
    -----------
    circuit : QuantumCircuit
        The quantum circuit that was executed
    ibm_job_results : Dict[str, int]
        The counts from IBM cloud execution for this circuit
    circuit_index : int
        Index of this circuit in the batch (for reference)
    shots : int
        Number of shots used (for validation/reference)
    noise_types : List[str]
        List of noise model types for custom backends (e.g., ["heron", "modsi1000", "pc3"])
    noise_param : float
        Noise parameter for custom backends (e.g., p=0.001)
    custom_backend : BackendV2, optional
        Custom backend to use (defaults to FakeTorino architecture)
    custom_noise_model : NoiseModel, optional
        Custom noise model to override default
    ibm_backend_name : str
        Name of IBM backend used (for metadata)
        
    Returns:
    --------
    Dict containing results from all three backends with execution statistics
    """
    
    results = {}
    
    # Ensure circuit has measurements (should already have them)
    if not any(instr.operation.name == 'measure' for instr in circuit.data):
        circuit_with_measurements = circuit.copy()
        if not circuit_with_measurements.cregs:
            circuit_with_measurements.add_register('c', circuit.num_qubits)
        circuit_with_measurements.measure_all()
        circuit = circuit_with_measurements
    
    # Use FakeTorino for transpilation consistency (or could use the actual IBM backend)
    fake_torino = FakeTorino()
    
    # Transpile circuit for comparison consistency
    common_transpiled = transpile(
        circuit,
        fake_torino,
        basis_gates=['sx', 'x', 'rz', 'cz', 'id'],
        optimization_level=1
    )
    
    print(f"\nComparing circuit {circuit_index} (depth: {common_transpiled.depth()})")
    
    # 1. Noiseless Simulator
    print("Running on noiseless simulator...")
    noiseless_simulator = AerSimulator()
    noiseless_job = noiseless_simulator.run(common_transpiled, shots=shots)
    noiseless_result = noiseless_job.result()
    
    results['noiseless'] = {
        'backend': 'AerSimulator (noiseless)',
        'counts': noiseless_result.get_counts(),
        'fidelity': 1.0,
        'execution_time': getattr(noiseless_result, 'time_taken', 'N/A')
    }
    
    # 2. IBM Cloud Real Hardware Results
    print(f"Using IBM cloud results from {ibm_backend_name}...")
    
    # Calculate fidelity relative to noiseless
    ibm_fidelity = _calculate_fidelity(
        results['noiseless']['counts'],
        ibm_job_results
    )
    
    results['ibm_hardware'] = {
        'backend': f'{ibm_backend_name} (IBM Cloud)',
        'counts': ibm_job_results,
        'fidelity': ibm_fidelity,
        'execution_time': 'N/A',  # Would need to extract from job metadata
        'circuit_index': circuit_index
    }
    
    # 3. Custom Backends with Multiple Noise Models
    # Set up backend
    if custom_backend is None:
        custom_backend = fake_torino
    
    # Set up QubitTracking
    qt = QubitTracking(custom_backend, common_transpiled)
    
    # Run each noise model
    for noise_type in noise_types:
        # Extract noise type and parameter for ModSI1000 variants
        if ':' in noise_type:
            base_type, p_str = noise_type.split(':')
            p_value = float(p_str)
        else:
            base_type = noise_type
            p_value = noise_param
            
        print(f"Running on custom backend with {noise_type} noise...")
        
        # Get noise model
        if base_type == "heron":
            noise_model = HeronNoise.get_noise(qt, custom_backend)
        elif base_type == "modsi1000":
            noise_model = ArtificialNoise.modSI1000(p_value, qt)
        elif base_type == "pc3":
            noise_model = ArtificialNoise.PC3(p_value, qt)
        else:
            noise_model = get_noise_model(base_type, qt, p_value, custom_backend)
        
        if custom_noise_model is not None:
            noise_model = custom_noise_model
        
        # Create Qiskit noise model approximation
        qiskit_custom_noise = _create_qiskit_noise_approximation(noise_model, custom_backend, use_cz=True)
        custom_simulator = AerSimulator(noise_model=qiskit_custom_noise)
        
        custom_job = custom_simulator.run(common_transpiled, shots=shots)
        custom_result = custom_job.result()
        
        # Calculate fidelity relative to noiseless
        custom_fidelity = _calculate_fidelity(
            results['noiseless']['counts'],
            custom_result.get_counts()
        )
        
        # Store results with noise type key
        results[f'{noise_type}_noise'] = {
            'backend': f'Custom backend with {noise_type} noise',
            'counts': custom_result.get_counts(),
            'fidelity': custom_fidelity,
            'execution_time': getattr(custom_result, 'time_taken', 'N/A'),
            'noise_parameters': {
                'sq_error': getattr(noise_model, 'sq', 'N/A'),
                'tq_error': getattr(noise_model, 'tq', 'N/A'),
                'measure_error': getattr(noise_model, 'measure', 'N/A'),
                'reset_error': getattr(noise_model, 'reset', 'N/A')
            }
        }
    
    # Maintain backward compatibility - if 'heron' is in noise_types, also set 'custom_noise'
    if 'heron' in noise_types:
        results['custom_noise'] = results['heron_noise']
    
    # Summary comparison
    results['comparison'] = _generate_comparison_summary_ibm(results)
    
    return results


def _generate_comparison_summary_ibm(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comparison summary for IBM cloud results."""
    summary = {
        'fidelities': {},
        'best_performing': None,
        'worst_performing': None,
        'analysis': {}
    }
    
    # Extract fidelities
    for backend_key in ['noiseless', 'ibm_hardware', 'custom_noise']:
        if backend_key in results:
            summary['fidelities'][backend_key] = results[backend_key]['fidelity']
    
    # Find best/worst performing (excluding noiseless which is always 1.0)
    noisy_fidelities = {k: v for k, v in summary['fidelities'].items() if k != 'noiseless'}
    if noisy_fidelities:
        summary['best_performing'] = max(noisy_fidelities.keys(), key=lambda k: noisy_fidelities[k])
        summary['worst_performing'] = min(noisy_fidelities.keys(), key=lambda k: noisy_fidelities[k])
    
    # Analysis
    if 'ibm_hardware' in summary['fidelities'] and 'custom_noise' in summary['fidelities']:
        ibm_fid = summary['fidelities']['ibm_hardware']
        custom_fid = summary['fidelities']['custom_noise']
        diff = abs(ibm_fid - custom_fid)
        
        summary['analysis'] = {
            'fidelity_difference': diff,
            'relative_error': diff / max(ibm_fid, custom_fid) if max(ibm_fid, custom_fid) > 0 else 0,
            'custom_vs_ibm': 'better' if custom_fid > ibm_fid else 'worse' if custom_fid < ibm_fid else 'equal'
        }
    
    return summary


def _calculate_success_probability(counts: Dict[str, int]) -> float:
    """Calculate success probability. For Bell states, success is '00' or '11'."""
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0
    
    # Get the bit string length
    if not counts:
        return 0.0
    
    sample_key = list(counts.keys())[0]
    
    # For Bell states (2 qubits), success is '00' or '11'
    if ' ' in sample_key:  # Format like '00 00' or '11 00'
        # For Bell state, we want correlated outcomes: both qubits same value
        success_states = ['00 00', '11 00']  # Perfect Bell state outcomes
        success_count = sum(counts.get(state, 0) for state in success_states)
    elif len(sample_key) == 2:  # Simple 2-qubit case without spaces
        # Bell state success: both qubits same value
        success_states = ['00', '11']
        success_count = sum(counts.get(state, 0) for state in success_states)
    else:
        # For other circuits, assume ground state is target
        ground_state = '0' * len(sample_key)
        success_count = counts.get(ground_state, 0)
    
    return success_count / total_shots


def _normalize_counts_format(counts: Dict[str, int]) -> Dict[str, int]:
    """
    Normalize count dictionary format to handle different classical register formats.
    
    IBM hardware returns: {'00': 100, '01': 200, ...}
    Simulators may return: {'00 00': 100, '01 00': 200, ...}
    
    This function extracts only the meaningful measurement bits.
    """
    if not counts:
        return counts
    
    # Check if we have space-separated registers
    sample_key = list(counts.keys())[0]
    if ' ' in sample_key:
        # Multiple classical registers - extract the non-zero register
        normalized = {}
        
        for bitstring, count in counts.items():
            parts = bitstring.split()
            
            # Find which register has varying bits (not all zeros)
            active_register = None
            for part in parts:
                if part != '0' * len(part):  # Not all zeros
                    active_register = part
                    break
            
            # If no active register found, use the first one
            if active_register is None:
                active_register = parts[0]
            
            normalized[active_register] = normalized.get(active_register, 0) + count
        
        return normalized
    
    else:
        # Single register format - return as is
        return counts


def _calculate_fidelity(ideal_counts: Dict[str, int], noisy_counts: Dict[str, int]) -> float:
    """Calculate fidelity between two count distributions using Total Variation distance.
    
    This is the proper metric for comparing random circuit results.
    Fidelity = 1 - 0.5 * sum(|p_ideal(x) - p_noisy(x)|)
    
    Args:
        ideal_counts: Counts from noiseless simulation
        noisy_counts: Counts from noisy simulation
        
    Returns:
        Fidelity between 0 and 1 (1 = identical distributions)
    """
    
    # Normalize both count dictionaries to handle different classical register formats
    ideal_normalized = _normalize_counts_format(ideal_counts)
    noisy_normalized = _normalize_counts_format(noisy_counts)
    
    # Get total shots for normalization
    total_ideal = sum(ideal_normalized.values())
    total_noisy = sum(noisy_normalized.values())
    
    if total_ideal == 0 or total_noisy == 0:
        return 0.0
    
    # Get all possible outcomes
    all_outcomes = set(ideal_normalized.keys()) | set(noisy_normalized.keys())
    
    # Calculate Total Variation distance
    tv_distance = 0.0
    for outcome in all_outcomes:
        p_ideal = ideal_normalized.get(outcome, 0) / total_ideal
        p_noisy = noisy_normalized.get(outcome, 0) / total_noisy
        tv_distance += abs(p_ideal - p_noisy)
    
    # Convert TV distance to fidelity
    fidelity = 1.0 - 0.5 * tv_distance
    return max(0.0, fidelity)  # Ensure non-negative


def _extract_torino_noise_parameters(torino_noise_model, fake_torino_backend):
    """Extract noise parameters from FakeTorino's noise model for comparison."""
    noise_details = {
        'gate_errors': {},
        'readout_errors': {},
        'reset_errors': {},
        't1_times': {},
        't2_times': {}
    }
    
    try:
        # Try to get backend properties from FakeTorino directly
        if hasattr(fake_torino_backend, 'properties'):
            props = fake_torino_backend.properties()
            
            # Single-qubit gate errors (average across qubits)
            sx_errors = []
            x_errors = []
            rz_errors = []
            
            for qubit in range(fake_torino_backend.num_qubits):
                try:
                    # Get single-qubit gate errors
                    if hasattr(props, 'gate_error'):
                        sx_error = props.gate_error('sx', qubit)
                        x_error = props.gate_error('x', qubit) 
                        rz_error = props.gate_error('rz', qubit)
                        
                        sx_errors.append(sx_error)
                        x_errors.append(x_error)
                        rz_errors.append(rz_error)
                except:
                    continue
            
            if sx_errors:
                noise_details['gate_errors']['sx_avg'] = np.mean(sx_errors)
                noise_details['gate_errors']['x_avg'] = np.mean(x_errors)
                noise_details['gate_errors']['rz_avg'] = np.mean(rz_errors)
            
            # Two-qubit gate errors (CZ and CX)
            cz_errors = []
            cx_errors = []
            coupling_map = fake_torino_backend.coupling_map
            if coupling_map:
                for edge in coupling_map.get_edges():
                    try:
                        # Try CZ first (Heron devices)
                        if hasattr(props, 'gate_error'):
                            try:
                                cz_error = props.gate_error('cz', edge)
                                cz_errors.append(cz_error)
                            except Exception as e:
                                pass
                            
                            try:
                                cx_error = props.gate_error('cx', edge)
                                cx_errors.append(cx_error)
                            except Exception as e:
                                pass
                    except:
                        continue
            
            # Filter out unrealistic errors (> 50% are likely disconnected qubits)
            if cz_errors:
                filtered_cz = [err for err in cz_errors if err < 0.5]  # Remove 100% error qubits
                if filtered_cz:
                    noise_details['gate_errors']['cz_avg'] = np.mean(filtered_cz)
                    
            if cx_errors:
                filtered_cx = [err for err in cx_errors if err < 0.5]  # Remove 100% error qubits
                if filtered_cx:
                    noise_details['gate_errors']['cx_avg'] = np.mean(filtered_cx)
            
            # Readout errors
            readout_errors = []
            for qubit in range(fake_torino_backend.num_qubits):
                try:
                    if hasattr(props, 'readout_error'):
                        ro_error = props.readout_error(qubit)
                        readout_errors.append(ro_error)
                except:
                    continue
            
            if readout_errors:
                noise_details['readout_errors']['average'] = np.mean(readout_errors)
            
            # Try to extract reset errors from the noise model
            if hasattr(torino_noise_model, '_default_quantum_errors'):
                errors = torino_noise_model._default_quantum_errors
                if 'reset' in errors:
                    reset_info = errors['reset']
                    if hasattr(reset_info, 'probabilities') and reset_info.probabilities:
                        # Reset error is typically the probability of incorrect reset
                        reset_prob = reset_info.probabilities[0] if reset_info.probabilities[0] else 0
                        noise_details['reset_errors']['average'] = reset_prob
                        
            # T1 and T2 times
            t1_times = []
            t2_times = []
            for qubit in range(fake_torino_backend.num_qubits):
                try:
                    t1 = props.t1(qubit)
                    t2 = props.t2(qubit)
                    if t1 is not None:
                        t1_times.append(t1)
                    if t2 is not None:
                        t2_times.append(t2)
                except:
                    continue
            
            if t1_times:
                noise_details['t1_times']['average'] = np.mean(t1_times)
            if t2_times:
                noise_details['t2_times']['average'] = np.mean(t2_times)
        
    except Exception as e:
        noise_details['extraction_error'] = f"Could not extract parameters: {e}"
        
        # Fallback: use typical IBM values for FakeTorino based on the device spec
        noise_details['gate_errors']['sx_avg'] = 0.0007    # From our extraction above
        noise_details['gate_errors']['x_avg'] = 0.0007     # From our extraction above
        noise_details['gate_errors']['rz_avg'] = 0.0       # Virtual Z gate
        noise_details['gate_errors']['cx_avg'] = 0.0108    # Typical IBM CNOT error for Torino-class
        noise_details['gate_errors']['cz_avg'] = 0.0108    # Typical IBM CZ error for Torino-class
        noise_details['readout_errors']['average'] = 0.047  # From our extraction above
        noise_details['reset_errors']['average'] = 0.002   # Typical IBM reset error
        noise_details['t1_times']['average'] = 174e-6      # From our extraction above  
        noise_details['t2_times']['average'] = 145e-6      # From our extraction above
    
    return noise_details


def print_noise_comparison(torino_details: Dict, heron_params: Dict):
    """Print side-by-side comparison of noise models."""
    print(f"{'Parameter':<25} {'FakeTorino':<20} {'Heron Model':<20} {'Ratio (H/T)':<15}")
    print("-" * 80)
    
    # Compare single-qubit errors
    torino_sq = torino_details.get('gate_errors', {}).get('sx_avg', 'N/A')
    heron_sq = heron_params.get('sq_error', 'N/A')
    ratio_sq = _calculate_ratio(heron_sq, torino_sq)
    print(f"{'Single-qubit error':<25} {_format_value(torino_sq):<20} {_format_value(heron_sq):<20} {ratio_sq:<15}")
    
    # Compare two-qubit errors (prefer CZ for Heron compatibility)
    torino_tq = torino_details.get('gate_errors', {}).get('cz_avg', 
                    torino_details.get('gate_errors', {}).get('cx_avg', 'N/A'))
    heron_tq = heron_params.get('tq_error', 'N/A')
    ratio_tq = _calculate_ratio(heron_tq, torino_tq)
    gate_name = 'CZ' if 'cz_avg' in torino_details.get('gate_errors', {}) else 'CX'
    print(f"{f'Two-qubit error ({gate_name})':<25} {_format_value(torino_tq):<20} {_format_value(heron_tq):<20} {ratio_tq:<15}")
    
    # Compare readout errors
    torino_ro = torino_details.get('readout_errors', {}).get('average', 'N/A')
    heron_ro = heron_params.get('measure_error', 'N/A') 
    ratio_ro = _calculate_ratio(heron_ro, torino_ro)
    print(f"{'Measurement error':<25} {_format_value(torino_ro):<20} {_format_value(heron_ro):<20} {ratio_ro:<15}")
    
    # Compare reset errors
    torino_reset = torino_details.get('reset_errors', {}).get('average', 'N/A')
    heron_reset = heron_params.get('reset_error', 'N/A')
    ratio_reset = _calculate_ratio(heron_reset, torino_reset)
    print(f"{'Reset error':<25} {_format_value(torino_reset):<20} {_format_value(heron_reset):<20} {ratio_reset:<15}")
    
    print()  # Empty line before coherence times
    # Show T1/T2 times if available
    if 't1_times' in torino_details and 'average' in torino_details['t1_times']:
        t1_val = torino_details['t1_times']['average']
        print(f"{'T1 time (μs)':<25} {t1_val*1e6:.1f}μs")
    
    if 't2_times' in torino_details and 'average' in torino_details['t2_times']:
        t2_val = torino_details['t2_times']['average']
        print(f"{'T2 time (μs)':<25} {t2_val*1e6:.1f}μs")


def _format_value(val):
    """Format numerical values for display."""
    if val == 'N/A' or val is None:
        return 'N/A'
    try:
        if isinstance(val, (int, float)):
            if val < 0.001:
                return f"{val:.2e}"
            else:
                return f"{val:.6f}"
        return str(val)
    except:
        return str(val)


def _calculate_ratio(val1, val2):
    """Calculate ratio between two values."""
    try:
        if val1 == 'N/A' or val2 == 'N/A' or val1 is None or val2 is None:
            return 'N/A'
        v1 = float(val1)
        v2 = float(val2)
        if v2 == 0:
            return 'N/A'
        ratio = v1 / v2
        return f"{ratio:.1f}x"
    except (ValueError, TypeError):
        return 'N/A'


def _create_qiskit_noise_approximation(custom_noise_model, backend, use_cz=False):
    """
    Create a simplified Qiskit noise model approximation of the custom noise model.
    This is a basic approximation since your noise models are Stim-based.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    
    noise_model = NoiseModel()
    
    # Add single-qubit depolarizing error
    if hasattr(custom_noise_model, 'sq') and custom_noise_model.sq > 0:
        sq_error = depolarizing_error(custom_noise_model.sq, 1)
        noise_model.add_all_qubit_quantum_error(sq_error, ['sx', 'x', 'rz'])
    
    # Add two-qubit depolarizing error (CZ for Heron, CX for others)
    if hasattr(custom_noise_model, 'tq') and custom_noise_model.tq > 0:
        tq_error = depolarizing_error(custom_noise_model.tq, 2)
        if use_cz:
            noise_model.add_all_qubit_quantum_error(tq_error, ['cz'])
        else:
            noise_model.add_all_qubit_quantum_error(tq_error, ['cx', 'cz'])
    
    # Add measurement error
    if hasattr(custom_noise_model, 'measure') and custom_noise_model.measure > 0:
        measure_error = ReadoutError([[1-custom_noise_model.measure, custom_noise_model.measure],
                                    [custom_noise_model.measure, 1-custom_noise_model.measure]])
        noise_model.add_all_qubit_readout_error(measure_error)
    
    # Add reset error if available
    if hasattr(custom_noise_model, 'reset') and custom_noise_model.reset > 0:
        reset_error = depolarizing_error(custom_noise_model.reset, 1)
        noise_model.add_all_qubit_quantum_error(reset_error, ['reset'])
    
    return noise_model


def _generate_comparison_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary comparing the different execution results."""
    summary = {
        'success_probabilities': {},
        'error_rates': {},
        'fidelity_comparison': {}
    }
    
    noiseless_fidelity = results['noiseless']['fidelity']
    
    for backend_key in ['noiseless', 'fake_torino', 'custom_noise']:
        if backend_key in results:
            fidelity = results[backend_key]['fidelity']
            summary['success_probabilities'][backend_key] = fidelity
            summary['error_rates'][backend_key] = 1 - fidelity
            
            # Calculate fidelity relative to noiseless case
            if noiseless_fidelity > 0:
                summary['fidelity_comparison'][backend_key] = fidelity / noiseless_fidelity
            else:
                summary['fidelity_comparison'][backend_key] = 0.0
    
    return summary


def print_comparison_results(results: Dict[str, Any]):
    """Pretty print the comparison results."""
    print("\n" + "="*60)
    print("QUANTUM CIRCUIT EXECUTION COMPARISON")
    print("="*60)
    
    for backend_key in ['noiseless', 'fake_torino', 'custom_noise']:
        if backend_key in results:
            result = results[backend_key]
            print(f"\n{result['backend'].upper()}:")
            print(f"  Fidelity: {result['fidelity']:.4f}")
            print(f"  Error Rate: {1-result['fidelity']:.4f}")
            print(f"  Top 3 outcomes: {dict(sorted(result['counts'].items(), key=lambda x: x[1], reverse=True)[:3])}")
            
            if 'noise_parameters' in result:
                print(f"  Noise Parameters:")
                for param, value in result['noise_parameters'].items():
                    print(f"    {param}: {value}")
            
            if 'torino_noise_details' in result:
                print(f"  FakeTorino Noise Details:")
                for param, value in result['torino_noise_details'].items():
                    print(f"    {param}: {value}")
    
    if 'comparison' in results:
        print(f"\nFIDELITY COMPARISON (relative to noiseless):")
        for backend, fidelity in results['comparison']['fidelity_comparison'].items():
            print(f"  {backend}: {fidelity:.4f}")
    
    # Add noise parameter comparison if both models available
    if 'fake_torino' in results and 'custom_noise' in results:
        if 'torino_noise_details' in results['fake_torino'] and 'noise_parameters' in results['custom_noise']:
            print(f"\n{'='*60}")
            print("NOISE MODEL COMPARISON")
            print("="*60)
            print_noise_comparison(
                results['fake_torino']['torino_noise_details'], 
                results['custom_noise']['noise_parameters']
            )


# Example usage function
def example_bell_state_comparison():
    """Example usage with a Bell state circuit."""
    # Create a Bell state circuit
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure_all()
    
    # Run comparison
    results = compare_circuit_execution(
        circuit=circuit,
        shots=10000,
        noise_type="heron",
        noise_param=0.001
    )
    
    # Print results
    print_comparison_results(results)
    
    return results


def generate_random_circuit_with_torino_gates(
    num_qubits: int,
    depth: int,
    seed: Optional[int] = None
) -> QuantumCircuit:
    """
    Generate a random circuit using only FakeTorino's native gate set.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits in the circuit
    depth : int  
        Circuit depth (number of gate layers)
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    QuantumCircuit with only FakeTorino-compatible gates
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # FakeTorino's native gate set (check target for exact gates)
    fake_torino = FakeTorino()
    
    # Common IBM gate set - use CZ for Heron compatibility
    single_qubit_gates = ['x', 'sx', 'rz', 'id']  # X, SX, RZ, Identity
    two_qubit_gates = ['cz']  # Use CZ instead of CX for Heron compatibility
    
    circuit = QuantumCircuit(num_qubits, num_qubits)
    
    for layer in range(depth):
        # Randomly choose qubits to apply gates to
        available_qubits = list(range(num_qubits))
        random.shuffle(available_qubits)
        
        used_qubits = set()
        
        # Apply some two-qubit gates first
        num_two_qubit = random.randint(0, min(2, num_qubits // 2))
        for _ in range(num_two_qubit):
            if len(available_qubits) < 2:
                break
                
            # Choose two qubits that aren't already used
            available_pairs = [(i, j) for i in available_qubits 
                             for j in available_qubits 
                             if i != j and i not in used_qubits and j not in used_qubits]
            
            if not available_pairs:
                break
                
            qubit1, qubit2 = random.choice(available_pairs)
            
            # Apply CZ gate (Heron-compatible)
            circuit.cz(qubit1, qubit2)
            used_qubits.update([qubit1, qubit2])
        
        # Apply single-qubit gates to remaining qubits
        remaining_qubits = [q for q in available_qubits if q not in used_qubits]
        for qubit in remaining_qubits:
            if random.random() < 0.7:  # 70% chance to apply a gate
                gate_type = random.choice(single_qubit_gates)
                
                if gate_type == 'x':
                    circuit.x(qubit)
                elif gate_type == 'sx':
                    circuit.sx(qubit)
                elif gate_type == 'rz':
                    # Random rotation angle
                    angle = random.uniform(0, 2 * np.pi)
                    circuit.rz(angle, qubit)
                elif gate_type == 'id':
                    circuit.id(qubit)
        
        # Add a barrier between layers for clarity
        if layer < depth - 1:
            circuit.barrier()
    
    # Add measurements
    circuit.measure_all()
    
    return circuit


def compare_random_circuits(
    num_qubits: int = 3,
    depth: int = 5,
    num_circuits: int = 5,
    shots: int = 1024,
    noise_type: str = "heron",
    noise_param: float = 0.001,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Generate multiple random circuits and compare their execution across backends.
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits per circuit
    depth : int
        Depth of each random circuit
    num_circuits : int
        Number of random circuits to test
    shots : int
        Shots per circuit execution
    noise_type : str
        Type of custom noise model
    noise_param : float
        Noise parameter
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    Dictionary with aggregated results across all circuits
    """
    
    print(f"\n{'='*60}")
    print(f"RANDOM CIRCUIT COMPARISON")
    print(f"Testing {num_circuits} random circuits ({num_qubits} qubits, depth {depth})")
    print(f"{'='*60}")
    
    all_results = {
        'noiseless': {'success_probs': [], 'avg_fidelity': 0},
        'fake_torino': {'success_probs': [], 'avg_fidelity': 0}, 
        'custom_noise': {'success_probs': [], 'avg_fidelity': 0},
        'circuits': []
    }
    
    for i in range(num_circuits):
        print(f"\n--- Circuit {i+1}/{num_circuits} ---")
        
        # Generate random circuit
        circuit = generate_random_circuit_with_torino_gates(
            num_qubits=num_qubits,
            depth=depth, 
            seed=seed + i  # Different seed for each circuit
        )
        
        all_results['circuits'].append(circuit)
        
        # Get ground state (all zeros) probability as success metric
        # For random circuits, we'll use ground state fidelity
        results = compare_circuit_execution(
            circuit=circuit,
            shots=shots,
            noise_type=noise_type,
            noise_param=noise_param
        )
        
        # Extract success probabilities  
        for backend_key in ['noiseless', 'fake_torino', 'custom_noise']:
            if backend_key in results:
                success_prob = _calculate_ground_state_fidelity(results[backend_key]['counts'])
                all_results[backend_key]['success_probs'].append(success_prob)
        
        # Print brief summary for this circuit
        print(f"  Noiseless: {all_results['noiseless']['success_probs'][-1]:.4f}")
        print(f"  FakeTorino: {all_results['fake_torino']['success_probs'][-1]:.4f}")  
        print(f"  Custom: {all_results['custom_noise']['success_probs'][-1]:.4f}")
    
    # Calculate averages
    for backend_key in ['noiseless', 'fake_torino', 'custom_noise']:
        if all_results[backend_key]['success_probs']:
            all_results[backend_key]['avg_fidelity'] = np.mean(all_results[backend_key]['success_probs'])
            all_results[backend_key]['std_fidelity'] = np.std(all_results[backend_key]['success_probs'])
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS {num_circuits} RANDOM CIRCUITS")
    print(f"{'='*60}")
    
    for backend_key in ['noiseless', 'fake_torino', 'custom_noise']:
        backend_name = {
            'noiseless': 'Noiseless',
            'fake_torino': 'FakeTorino', 
            'custom_noise': f'Custom ({noise_type})'
        }[backend_key]
        
        if backend_key in all_results and all_results[backend_key]['success_probs']:
            avg = all_results[backend_key]['avg_fidelity']
            std = all_results[backend_key]['std_fidelity']
            print(f"{backend_name}: {avg:.4f} ± {std:.4f}")
    
    return all_results


def compare_random_circuits_with_ibm_cloud(
    num_qubits: int = 3,
    depth: int = 5,
    num_circuits: int = None,  # Will default to 10 * num_qubits
    shots: int = 1000,
    noise_type: str = "heron",
    noise_param: float = 0.001,
    seed: int = 42,
    backend_name: str = None,
    submit_only: bool = False,
    job_id: str = None
) -> Dict[str, Any]:
    """
    Generate random circuits and compare execution between IBM cloud and custom noise model.
    
    This function can work in two modes:
    1. Submit mode (submit_only=True): Generate circuits and submit to IBM cloud
    2. Analysis mode (job_id provided): Retrieve results and perform comparison
    
    Parameters:
    -----------
    num_qubits : int
        Number of qubits per circuit
    depth : int 
        Depth of each random circuit
    num_circuits : int, optional
        Number of random circuits to test (defaults to 10 * num_qubits)
    shots : int
        Shots per circuit execution (default: 1000)
    noise_type : str
        Type of custom noise model for comparison
    noise_param : float
        Noise parameter
    seed : int
        Random seed for reproducibility
    backend_name : str, optional
        IBM backend name (auto-selected if None)
    submit_only : bool
        If True, only submit job and return job info
    job_id : str, optional
        If provided, retrieve results from this job instead of submitting
        
    Returns:
    --------
    Dict with job info (submit mode) or comparison results (analysis mode)
    """
    
    if job_id is not None:
        # Analysis mode: retrieve and process existing job
        return _process_ibm_job_results(
            job_id=job_id,
            noise_type=noise_type,
            noise_param=noise_param
        )
    
    # Submit mode: generate circuits and submit
    
    # Set default num_circuits to 10 per qubit if not specified
    if num_circuits is None:
        num_circuits = num_qubits * 10
    
    print(f"\n{'='*60}")
    print(f"RANDOM CIRCUIT IBM CLOUD COMPARISON")
    print(f"Generating {num_circuits} random circuits ({num_qubits} qubits, depth {depth})")
    print(f"Using 10 circuits per qubit (configurable)")
    print(f"Seed: {seed}")
    print(f"{'='*60}")
    
    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate random circuits with fixed seed
    circuits = []
    for i in range(num_circuits):
        circuit = generate_random_circuit_with_torino_gates(
            num_qubits=num_qubits,
            depth=depth,
            seed=seed + i  # Different seed for each circuit but deterministic
        )
        circuits.append(circuit)
        print(f"Generated circuit {i+1}: depth={circuit.depth()}, size={circuit.size()}")
    
    # Select IBM backend if not specified
    if backend_name is None:
        backend_name = select_best_ibm_backend(min_qubits=num_qubits)
        if backend_name is None:
            raise RuntimeError("No suitable IBM backend available")
    
    # Estimate job cost
    cost_estimate = estimate_job_cost(num_circuits, shots, backend_name)
    print(f"\nJob cost estimate:")
    print(f"  Total shots: {cost_estimate['total_shots']}")
    print(f"  Est. execution time: {cost_estimate['estimated_execution_time_minutes']:.1f} min")
    for rec in cost_estimate['recommendation']:
        print(f"  Recommendation: {rec}")
    
    # Submit to IBM cloud
    job_info = submit_circuits_to_ibm_cloud(
        circuits=circuits,
        backend_name=backend_name,
        shots_per_circuit=shots,
        job_name=f"random_circuits_{num_circuits}x{num_qubits}q_seed{seed}"
    )
    
    if submit_only:
        print(f"\nJob submitted! Use job_id '{job_info['job_id']}' to retrieve results later.")
        print(f"To analyze results: compare_random_circuits_with_ibm_cloud(job_id='{job_info['job_id']}')")
        return {
            'mode': 'submit',
            'job_info': job_info,
            'circuits': circuits,  # Store for later reference
            'parameters': {
                'num_qubits': num_qubits,
                'depth': depth,
                'num_circuits': num_circuits,
                'shots': shots,
                'seed': seed,
                'noise_type': noise_type,
                'noise_param': noise_param
            }
        }
    
    # If not submit_only, wait for results and analyze
    print(f"\nWaiting for IBM job completion...")
    ibm_results = get_ibm_job_results(job_info['job_id'], wait_for_completion=True)
    
    if 'error' in ibm_results:
        print(f"Error retrieving results: {ibm_results['error']}")
        return {'error': ibm_results['error'], 'job_info': job_info}
    
    # Process results
    return _process_ibm_job_results(
        job_id=job_info['job_id'],
        ibm_results=ibm_results,
        circuits=circuits,
        noise_type=noise_type,
        noise_param=noise_param
    )


def _process_ibm_job_results(
    job_id: str,
    noise_type: str = "heron",
    noise_param: float = 0.001,
    ibm_results: Dict = None,
    circuits: List[QuantumCircuit] = None
) -> Dict[str, Any]:
    """Process IBM job results and compare with custom noise model."""
    
    # Retrieve results if not provided
    if ibm_results is None:
        print(f"Retrieving results for job {job_id}...")
        ibm_results = get_ibm_job_results(job_id)
        
        if 'error' in ibm_results:
            print(f"Error retrieving results: {ibm_results['error']}")
            return {'error': ibm_results['error']}
    
    # Try to load job info and circuits if not provided
    if circuits is None:
        job_info = load_job_info_from_file(job_id)
        if 'circuit_qasm' in job_info:
            circuits = [QuantumCircuit.from_qasm_str(qasm) for qasm in job_info['circuit_qasm']]
        else:
            print("Warning: Could not load original circuits. Analysis will be limited.")
            circuits = []
    
    counts_list = ibm_results['counts']
    backend_name = ibm_results['metadata'].get('backend_name', 'unknown')
    
    print(f"\n{'='*60}")
    print(f"PROCESSING IBM CLOUD RESULTS")
    print(f"Job ID: {job_id}")
    print(f"Backend: {backend_name}")
    print(f"Circuits: {len(counts_list)}")
    print(f"{'='*60}")
    
    # Analyze each circuit
    all_results = {
        'noiseless': {'success_probs': [], 'avg_fidelity': 0},
        'ibm_hardware': {'success_probs': [], 'avg_fidelity': 0},
        'custom_noise': {'success_probs': [], 'avg_fidelity': 0},
        'circuits': circuits,
        'job_metadata': ibm_results['metadata']
    }
    
    for i, ibm_counts in enumerate(counts_list):
        print(f"\n--- Circuit {i+1}/{len(counts_list)} ---")
        
        if i < len(circuits):
            circuit = circuits[i]
            # Compare this circuit's execution
            comparison_results = compare_circuit_execution_with_ibm_cloud(
                circuit=circuit,
                ibm_job_results=ibm_counts,
                circuit_index=i,
                noise_type=noise_type,
                noise_param=noise_param,
                ibm_backend_name=backend_name
            )
            
            # Extract success probabilities (ground state fidelities)
            for backend_key in ['noiseless', 'ibm_hardware', 'custom_noise']:
                if backend_key in comparison_results:
                    success_prob = _calculate_ground_state_fidelity(comparison_results[backend_key]['counts'])
                    all_results[backend_key]['success_probs'].append(success_prob)
            
            print(f"  Noiseless: {all_results['noiseless']['success_probs'][-1]:.4f}")
            print(f"  IBM Hardware: {all_results['ibm_hardware']['success_probs'][-1]:.4f}")
            print(f"  Custom Noise: {all_results['custom_noise']['success_probs'][-1]:.4f}")
        
        else:
            print(f"  Warning: No circuit available for index {i}")
            # Just analyze IBM results vs ground state
            ibm_success = _calculate_ground_state_fidelity(ibm_counts)
            all_results['ibm_hardware']['success_probs'].append(ibm_success)
            print(f"  IBM Hardware: {ibm_success:.4f}")
    
    # Calculate summary statistics
    for backend_key in ['noiseless', 'ibm_hardware', 'custom_noise']:
        if all_results[backend_key]['success_probs']:
            all_results[backend_key]['avg_fidelity'] = np.mean(all_results[backend_key]['success_probs'])
            all_results[backend_key]['std_fidelity'] = np.std(all_results[backend_key]['success_probs'])
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"SUMMARY ACROSS {len(counts_list)} CIRCUITS")
    print(f"{'='*60}")
    
    backend_names = {
        'noiseless': 'Noiseless',
        'ibm_hardware': f'IBM {backend_name}',
        'custom_noise': f'Custom ({noise_type})'
    }
    
    for backend_key in ['noiseless', 'ibm_hardware', 'custom_noise']:
        if all_results[backend_key]['success_probs']:
            avg = all_results[backend_key]['avg_fidelity']
            std = all_results[backend_key]['std_fidelity']
            name = backend_names[backend_key]
            print(f"{name}: {avg:.4f} ± {std:.4f}")
    
    # Add comparison analysis
    if all_results['ibm_hardware']['success_probs'] and all_results['custom_noise']['success_probs']:
        ibm_avg = all_results['ibm_hardware']['avg_fidelity']
        custom_avg = all_results['custom_noise']['avg_fidelity']
        
        print(f"\nModel Comparison:")
        print(f"  Custom model {'overestimates' if custom_avg > ibm_avg else 'underestimates'} hardware performance")
        print(f"  Difference: {abs(custom_avg - ibm_avg):.4f}")
        print(f"  Relative error: {abs(custom_avg - ibm_avg) / ibm_avg * 100:.1f}%")
    
    return all_results


def run_full_noise_experiment_ibm_cloud(
    qubit_range: Tuple[int, int] = (2, 7),
    circuits_per_qubit_size: int = 10,
    circuit_depth: int = 100,
    shots: int = 1000,
    seed: int = 42,
    backend_name: str = None,
    submit_only: bool = False,
    job_id: str = None
) -> Dict[str, Any]:
    """
    Run the complete noise model experiment: 60 circuits across qubit sizes 2-7.
    
    This creates 10 circuits each for qubit sizes 2, 3, 4, 5, 6, 7 (60 total circuits)
    with depth 100, comparing IBM hardware vs. Heron noise model.
    
    Parameters:
    -----------
    qubit_range : Tuple[int, int]
        Range of qubit sizes to test (default: 2 to 7 inclusive)
    circuits_per_qubit_size : int
        Number of circuits per qubit size (default: 10)
    circuit_depth : int
        Depth of each circuit (default: 100)
    shots : int
        Shots per circuit (default: 1000)
    seed : int
        Random seed for reproducibility
    backend_name : str, optional
        IBM backend name (auto-selected if None)
    submit_only : bool
        If True, only submit job and return job info
    job_id : str, optional
        If provided, analyze results from existing job
        
    Returns:
    --------
    Dict with job info (submit mode) or complete experimental results (analysis mode)
    """
    
    if job_id is not None:
        # Analysis mode: retrieve and process existing job
        return _process_full_experiment_results(job_id=job_id)
    
    # Calculate total circuits
    qubit_sizes = list(range(qubit_range[0], qubit_range[1] + 1))
    total_circuits = len(qubit_sizes) * circuits_per_qubit_size
    total_shots = total_circuits * shots
    
    print(f"\n{'='*70}")
    print(f"FULL NOISE MODEL EXPERIMENT - IBM CLOUD")
    print(f"{'='*70}")
    print(f"Qubit sizes: {qubit_sizes}")
    print(f"Circuits per qubit size: {circuits_per_qubit_size}")
    print(f"Circuit depth: {circuit_depth}")
    print(f"Total circuits: {total_circuits}")
    print(f"Shots per circuit: {shots}")
    print(f"Total shots: {total_shots:,}")
    print(f"Seed: {seed}")
    print(f"{'='*70}")
    
    # Set the random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Generate all circuits
    all_circuits = []
    circuit_metadata = []
    
    for qubit_size in qubit_sizes:
        print(f"\nGenerating {circuits_per_qubit_size} circuits with {qubit_size} qubits...")
        for i in range(circuits_per_qubit_size):
            circuit = generate_random_circuit_with_torino_gates(
                num_qubits=qubit_size,
                depth=circuit_depth,
                seed=seed + qubit_size * 1000 + i  # Ensure unique but deterministic seeds
            )
            all_circuits.append(circuit)
            circuit_metadata.append({
                'qubit_size': qubit_size,
                'circuit_index': i,
                'depth': circuit.depth(),
                'size': circuit.size()
            })
        
        avg_depth = np.mean([c.depth() for c in all_circuits[-circuits_per_qubit_size:]])
        avg_size = np.mean([c.size() for c in all_circuits[-circuits_per_qubit_size:]])
        print(f"  {qubit_size}-qubit circuits: avg depth={avg_depth:.1f}, avg size={avg_size:.1f}")
    
    print(f"\n✓ Generated {len(all_circuits)} circuits total")
    
    # Select IBM backend if not specified
    if backend_name is None:
        backend_name = select_best_ibm_backend(min_qubits=max(qubit_sizes))
        if backend_name is None:
            raise RuntimeError(f"No suitable IBM backend available for {max(qubit_sizes)} qubits")
    
    # Estimate cost
    print(f"\nEstimated resource usage:")
    print(f"  Backend: {backend_name}")
    print(f"  Total shots: {total_shots:,}")
    print(f"  Estimated queue time: depends on current load")
    print(f"  Estimated execution time: {total_shots / 1000:.1f} minutes")
    
    # Submit to IBM cloud
    job_info = submit_circuits_to_ibm_cloud(
        circuits=all_circuits,
        backend_name=backend_name,
        shots_per_circuit=shots,
        job_name=f"noise_experiment_60circuits_depth{circuit_depth}_seed{seed}"
    )
    
    # Add experiment metadata to job info
    job_info['experiment_metadata'] = {
        'qubit_range': qubit_range,
        'circuits_per_qubit_size': circuits_per_qubit_size,
        'circuit_depth': circuit_depth,
        'circuit_metadata': circuit_metadata,
        'qubit_sizes': qubit_sizes
    }
    
    if submit_only:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT SUBMITTED TO IBM CLOUD")
        print(f"{'='*70}")
        print(f"Job ID: {job_info['job_id']}")
        print(f"Backend: {job_info['backend_name']}")
        print(f"Total circuits: {total_circuits}")
        print(f"Total shots: {total_shots:,}")
        print(f"\nTo analyze results later:")
        print(f"  run_full_noise_experiment_ibm_cloud(job_id='{job_info['job_id']}')")
        print(f"  # OR")
        print(f"  python ibm_demo.py analyze {job_info['job_id']}")
        
        return {
            'mode': 'submit',
            'job_info': job_info,
            'circuits': all_circuits,
            'circuit_metadata': circuit_metadata,
            'experiment_parameters': {
                'qubit_range': qubit_range,
                'circuits_per_qubit_size': circuits_per_qubit_size,
                'circuit_depth': circuit_depth,
                'shots': shots,
                'seed': seed
            }
        }
    
    # If not submit_only, wait for results and analyze
    print(f"\nWaiting for IBM job completion...")
    print(f"This may take a while for {total_circuits} circuits...")
    
    ibm_results = get_ibm_job_results(job_info['job_id'], wait_for_completion=True)
    
    if 'error' in ibm_results:
        print(f"Error retrieving results: {ibm_results['error']}")
        return {'error': ibm_results['error'], 'job_info': job_info}
    
    # Process results
    return _process_full_experiment_results(
        job_id=job_info['job_id'],
        ibm_results=ibm_results,
        circuits=all_circuits,
        circuit_metadata=circuit_metadata
    )


def _process_full_experiment_results(
    job_id: str,
    ibm_results: Dict = None,
    circuits: List[QuantumCircuit] = None,
    circuit_metadata: List[Dict] = None
) -> Dict[str, Any]:
    """Process the full 60-circuit experiment results."""
    
    # Retrieve results if not provided
    if ibm_results is None:
        print(f"Retrieving results for experiment job {job_id}...")
        ibm_results = get_ibm_job_results(job_id)
        
        if 'error' in ibm_results:
            print(f"Error retrieving results: {ibm_results['error']}")
            return {'error': ibm_results['error']}
    
    # Try to load job info if circuits not provided
    if circuits is None or circuit_metadata is None:
        job_info = load_job_info_from_file(job_id)
        if 'circuit_qasm' in job_info:
            try:
                circuits = [QuantumCircuit.from_qasm_str(qasm) for qasm in job_info['circuit_qasm']]
            except Exception as e:
                print(f"Warning: Could not load circuits from QASM: {e}")
                circuits = []
        if 'experiment_metadata' in job_info:
            circuit_metadata = job_info['experiment_metadata']['circuit_metadata']
        else:
            print("Warning: No circuit metadata found in job info.")
            circuit_metadata = None
    
    # If we still don't have circuits, recreate them from the standard experimental parameters
    if not circuits or not circuit_metadata:
        print("Recreating original circuits using standard experimental parameters...")
        circuits, circuit_metadata = _recreate_experimental_circuits(
            qubit_range=(2, 7),
            circuits_per_qubit_size=10,
            circuit_depth=100,
            seed=42  # Standard seed used in experiment
        )
        print(f"✓ Recreated {len(circuits)} circuits")
    
    counts_list = ibm_results['counts']
    backend_name = ibm_results['metadata'].get('backend_name', 'unknown')
    
    print(f"\n{'='*70}")
    print(f"PROCESSING FULL EXPERIMENT RESULTS")
    print(f"{'='*70}")
    print(f"Job ID: {job_id}")
    print(f"Backend: {backend_name}")
    print(f"Total circuits: {len(counts_list)}")
    print(f"{'='*70}")
    
    # Organize results by qubit size
    results_by_qubit_size = {}
    overall_results = {
        'noiseless': {'fidelities': [], 'avg_fidelity': 0},
        'ibm_hardware': {'fidelities': [], 'avg_fidelity': 0},
        'heron_model': {'fidelities': [], 'avg_fidelity': 0},
        'modsi1000_001_model': {'fidelities': [], 'avg_fidelity': 0},
        'modsi1000_002_model': {'fidelities': [], 'avg_fidelity': 0},
        'job_metadata': ibm_results['metadata']
    }
    
    for i, ibm_counts in enumerate(counts_list):
        if i < len(circuits) and circuit_metadata and i < len(circuit_metadata):
            # Full analysis with original circuits
            circuit = circuits[i]
            metadata = circuit_metadata[i]
            qubit_size = metadata['qubit_size']
            
            print(f"Analyzing circuit {i+1}/{len(counts_list)}: {qubit_size} qubits (full comparison)")
            
            # Initialize qubit size results if needed
            if qubit_size not in results_by_qubit_size:
                results_by_qubit_size[qubit_size] = {
                    'noiseless': {'fidelities': []},
                    'ibm_hardware': {'fidelities': []},
                    'heron_model': {'fidelities': []},
                    'modsi1000_001_model': {'fidelities': []},
                    'modsi1000_002_model': {'fidelities': []},
                    'circuits': []
                }
            
            # Compare this circuit's execution (using multiple noise models)
            comparison_results = compare_circuit_execution_with_ibm_cloud(
                circuit=circuit,
                ibm_job_results=ibm_counts,
                circuit_index=i,
                noise_types=["heron", "modsi1000:0.001", "modsi1000:0.002"],
                noise_param=0.001,
                ibm_backend_name=backend_name
            )
            
            # Extract fidelities - use proper calculation
            for backend_key in ['noiseless', 'ibm_hardware', 'heron_noise', 'modsi1000:0.001_noise', 'modsi1000:0.002_noise']:
                if backend_key in comparison_results:
                    if backend_key == 'noiseless':
                        # Noiseless is always 1.0 by definition
                        fidelity = 1.0
                    else:
                        # Calculate fidelity relative to noiseless
                        fidelity = _calculate_fidelity(
                            comparison_results['noiseless']['counts'],
                            comparison_results[backend_key]['counts']
                        )
                    
                    # Debugging: print some details for first few circuits
                    if i < 3:
                        print(f"    {backend_key}: fidelity = {fidelity:.4f}")
                        if backend_key in ['ibm_hardware', 'heron_noise', 'modsi1000:0.001_noise', 'modsi1000:0.002_noise']:
                            # Show some sample counts
                            counts = comparison_results[backend_key]['counts']
                            normalized = _normalize_counts_format(counts)
                            total = sum(counts.values())
                            top_outcomes = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
                            top_normalized = sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:3]
                            print(f"      Raw outcomes: {[(outcome, count/total) for outcome, count in top_outcomes]}")
                            print(f"      Normalized: {[(outcome, count/sum(normalized.values())) for outcome, count in top_normalized]}")
                    
                    # Map to our naming convention
                    if backend_key == 'heron_noise':
                        result_key = 'heron_model'
                    elif backend_key == 'modsi1000:0.001_noise':
                        result_key = 'modsi1000_001_model'
                    elif backend_key == 'modsi1000:0.002_noise':
                        result_key = 'modsi1000_002_model'
                    else:
                        result_key = backend_key
                    
                    results_by_qubit_size[qubit_size][result_key]['fidelities'].append(fidelity)
                    overall_results[result_key]['fidelities'].append(fidelity)
            
            results_by_qubit_size[qubit_size]['circuits'].append(circuit)
        
        else:
            # This shouldn't happen now that we recreate circuits, but keep as fallback
            print(f"Warning: Circuit {i+1}/{len(counts_list)}: falling back to limited analysis")
            
            # Just analyze IBM results vs ground state
            ibm_fidelity = _calculate_ground_state_fidelity(ibm_counts)
            overall_results['ibm_hardware']['fidelities'].append(ibm_fidelity)
            
            print(f"  IBM Hardware ground state prob: {ibm_fidelity:.4f}")
    
    # Calculate summary statistics
    print(f"\n{'='*70}")
    print(f"EXPERIMENTAL RESULTS SUMMARY")
    print(f"{'='*70}")
    
    for qubit_size in sorted(results_by_qubit_size.keys()):
        results = results_by_qubit_size[qubit_size]
        
        print(f"\n{qubit_size}-QUBIT CIRCUITS:")
        print(f"{'Backend':<15} {'Avg Fidelity':<12} {'Std Dev':<10} {'Circuits'}")
        print(f"{'-'*50}")
        
        for backend_key in ['noiseless', 'ibm_hardware', 'heron_model', 'modsi1000_001_model', 'modsi1000_002_model']:
            fidelities = results[backend_key]['fidelities']
            if fidelities:
                avg_fid = np.mean(fidelities)
                std_fid = np.std(fidelities)
                results[backend_key]['avg_fidelity'] = avg_fid
                results[backend_key]['std_fidelity'] = std_fid
                
                backend_name_map = {
                    'noiseless': 'Noiseless',
                    'ibm_hardware': f'IBM Hardware',
                    'heron_model': 'Heron Model',
                    'modsi1000_001_model': 'ModSI1000 (p=0.001)',
                    'modsi1000_002_model': 'ModSI1000 (p=0.002)'
                }
                
                print(f"{backend_name_map[backend_key]:<19} {avg_fid:<12.4f} {std_fid:<10.4f} {len(fidelities)}")
    
    # Overall statistics
    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS (ALL 60 CIRCUITS)")
    print(f"{'='*70}")
    
    for backend_key in ['noiseless', 'ibm_hardware', 'heron_model', 'modsi1000_001_model', 'modsi1000_002_model']:
        fidelities = overall_results[backend_key]['fidelities']
        if fidelities:
            overall_results[backend_key]['avg_fidelity'] = np.mean(fidelities)
            overall_results[backend_key]['std_fidelity'] = np.std(fidelities)
    
    # Model performance analysis
    if overall_results['ibm_hardware']['fidelities']:
        ibm_avg = overall_results['ibm_hardware']['avg_fidelity']
        ibm_std = overall_results['ibm_hardware']['std_fidelity']
        
        print(f"\nMODEL PERFORMANCE ANALYSIS:")
        print(f"  IBM Hardware: {ibm_avg:.4f} ± {ibm_std:.4f}")
        
        # Compare each noise model against IBM hardware
        for model_key in ['heron_model', 'modsi1000_001_model', 'modsi1000_002_model']:
            if overall_results[model_key]['fidelities']:
                model_avg = overall_results[model_key]['avg_fidelity']
                model_std = overall_results[model_key]['std_fidelity']
                diff = model_avg - ibm_avg
                rel_error = abs(diff) / ibm_avg * 100
                
                model_names = {
                    'heron_model': 'Heron Model',
                    'modsi1000_001_model': 'ModSI1000 (p=0.001)',
                    'modsi1000_002_model': 'ModSI1000 (p=0.002)'
                }
                
                print(f"  {model_names[model_key]}: {model_avg:.4f} ± {model_std:.4f}")
                print(f"    Difference: {diff:+.4f} ({rel_error:.1f}% error)")
                
                if diff > 0.01:
                    print(f"    → {model_names[model_key]} OVERESTIMATES hardware performance")
                elif diff < -0.01:
                    print(f"    → {model_names[model_key]} UNDERESTIMATES hardware performance")
                else:
                    print(f"    → {model_names[model_key]} closely matches hardware performance!")
    
    return {
        'results_by_qubit_size': results_by_qubit_size,
        'overall_results': overall_results,
        'job_metadata': ibm_results['metadata']
    }


def _recreate_experimental_circuits(
    qubit_range: Tuple[int, int] = (2, 7),
    circuits_per_qubit_size: int = 10,
    circuit_depth: int = 100,
    seed: int = 42
) -> Tuple[List[QuantumCircuit], List[Dict]]:
    """
    Recreate the original experimental circuits using the same parameters.
    
    This generates exactly the same circuits that were originally submitted
    by using the same deterministic seed sequence.
    """
    
    print(f"Recreating circuits: qubits {qubit_range[0]}-{qubit_range[1]}, "
          f"{circuits_per_qubit_size} per size, depth {circuit_depth}, seed {seed}")
    
    # Set the same random seed as used in original generation
    random.seed(seed)
    np.random.seed(seed)
    
    qubit_sizes = list(range(qubit_range[0], qubit_range[1] + 1))
    circuits = []
    circuit_metadata = []
    
    for qubit_size in qubit_sizes:
        for i in range(circuits_per_qubit_size):
            # Use the same seed pattern as in the original function
            circuit_seed = seed + qubit_size * 1000 + i
            
            circuit = generate_random_circuit_with_torino_gates(
                num_qubits=qubit_size,
                depth=circuit_depth,
                seed=circuit_seed
            )
            
            circuits.append(circuit)
            circuit_metadata.append({
                'qubit_size': qubit_size,
                'circuit_index': i,
                'depth': circuit.depth(),
                'size': circuit.size()
            })
    
    print(f"✓ Generated {len(circuits)} circuits matching original experiment")
    return circuits, circuit_metadata


def plot_experimental_results(results: Dict[str, Any], save_path: str = None) -> str:
    """Plot the experimental results comparing IBM hardware vs all noise models."""
    
    if 'results_by_qubit_size' not in results:
        print("No qubit-wise results found for plotting")
        return None
    
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Apply the same rcParams settings as in plots/utils.py
    tex_fonts = {
        "font.family": "serif",
        "axes.labelsize": 12,
        "font.size": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.titlesize": 10,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "lines.markeredgewidth": 1.5,
        "lines.markeredgecolor": "black",
        "errorbar.capsize": 3,
    }
    plt.rcParams.update(tex_fonts)
    
    # Extract data for plotting (only qubits 3-7)
    qubit_sizes = sorted([q for q in results['results_by_qubit_size'].keys() if q >= 3])
    
    # Model configurations using seaborn pastel palette like in plots.py
    palette = sns.color_palette("pastel", n_colors=4)
    models = {
        'ibm_hardware': {'label': 'IBM Torino', 'color': palette[0], 'alpha': 1.0},
        'heron_model': {'label': 'Heron Model', 'color': palette[1], 'alpha': 1.0},
        'modsi1000_001_model': {'label': 'SI1000 (p=0.001)', 'color': palette[2], 'alpha': 1.0},
        'modsi1000_002_model': {'label': 'SI1000 (p=0.002)', 'color': palette[3], 'alpha': 1.0}
    }
    
    # Prepare data arrays
    data = {model: {'means': [], 'stds': []} for model in models}
    
    for qubits in qubit_sizes:
        qubit_data = results['results_by_qubit_size'][qubits]
        
        for model_key in models:
            if model_key in qubit_data and qubit_data[model_key]['fidelities']:
                data[model_key]['means'].append(qubit_data[model_key].get('avg_fidelity', 0))
                data[model_key]['stds'].append(qubit_data[model_key].get('std_fidelity', 0))
            else:
                data[model_key]['means'].append(0)
                data[model_key]['stds'].append(0)
    
    # Create the plot with the same dimensions as their plots
    fig, ax = plt.subplots(figsize=(5, 2.4))  # Using their HEIGHT_FIGSIZE * 2
    
    # Add blue border around the entire figure
    fig.patch.set_edgecolor('blue')
    fig.patch.set_linewidth(3)
    
    x = np.array(qubit_sizes)
    n_models = len(models)
    bar_width = 0.15  # Match their BAR_WIDTH style
    
    # Create bars for each model with their styling
    bars = {}
    for i, (model_key, config) in enumerate(models.items()):
        offset = (i - (n_models-1)/2) * bar_width
        bars[model_key] = ax.bar(x + offset, data[model_key]['means'], bar_width, 
                                yerr=data[model_key]['stds'], 
                                label=config['label'], 
                                alpha=config['alpha'], 
                                capsize=3,
                                color=config['color'],
                                edgecolor='black',  # Black edges like their plots
                                linewidth=1)
    
    # Customize the plot to match their style
    ax.set_xlabel('Circuit Width', fontsize=12)
    ax.set_ylabel('Fidelity', fontsize=12)
    ax.set_title('Noise Comparison', 
                 fontsize=12, fontweight='bold', loc='left')  # Left-aligned title like theirs
    ax.set_xticks(x)
    ax.set_xticklabels(qubit_sizes)
    ax.legend(fontsize=10, loc='lower left', ncol=2)
    ax.grid(axis='y', alpha=0.3)  # Only y-grid like their plots
    ax.set_axisbelow(True)  # Grid behind bars
    ax.set_ylim(0, 1.0)
    
    # Ensure axes spines are black like their plots
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)
    
    # Add value labels on bars (show values for all models)
    #def add_value_labels(bars, means, stds, model_name):
    #    for bar, mean, std in zip(bars, means, stds):
    #        if mean > 0:  # Only add label if there's data
    #            height = bar.get_height()
    #            # Use smaller font and different styling for different models
    #            if model_name == 'IBM Hardware':
    #                fontsize = 8
    #                fontweight = 'bold'
    #                color = 'black'
    #            else:
    #                fontsize = 7
    #                fontweight = 'normal'  
    #                color = 'darkblue'
    #            
    #            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.005,
    #                   f'{mean:.3f}',
    #                   ha='center', va='bottom', fontsize=fontsize, 
    #                   fontweight=fontweight, color=color)
    
    #for model_key, config in models.items():
    #    add_value_labels(bars[model_key], data[model_key]['means'], 
    #                    data[model_key]['stds'], config['label'])
    
    # Add overall statistics as text
    #if 'overall_results' in results:
    #    overall = results['overall_results']
    #    textstr = 'Overall Fidelity (All 60 circuits):\\n'
    #    
    #    for model_key, config in models.items():
    #        if (model_key in overall and overall[model_key]['fidelities']):
    #            avg = overall[model_key]['avg_fidelity']
    #            std = overall[model_key]['std_fidelity']
    #            textstr += f'{config["label"]}: {avg:.3f} ± {std:.3f}\\n'
    #    
    #    # Add text box
    #    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    #    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
    #           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"experimental_results_multi_model_{timestamp}.pdf"
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.show()
    
    print(f"Plot saved to: {save_path}")
    return save_path


def _calculate_ground_state_fidelity(counts: Dict[str, int]) -> float:
    """Calculate fidelity to ground state (all zeros)."""
    
    # Normalize counts first to handle different register formats
    normalized_counts = _normalize_counts_format(counts)
    
    total_shots = sum(normalized_counts.values())
    if total_shots == 0:
        return 0.0
    
    # Look for ground state pattern (all zeros)
    if normalized_counts:
        sample_key = list(normalized_counts.keys())[0]
        ground_pattern = '0' * len(sample_key)
    else:
        return 0.0
    
    ground_count = normalized_counts.get(ground_pattern, 0)
    return ground_count / total_shots


# Example usage for random circuits
def example_random_circuit_comparison():
    """Example usage with random circuits."""
    results = compare_random_circuits(
        num_qubits=3,
        depth=4,
        num_circuits=3,
        shots=1024,
        noise_type="heron",
        seed=42
    )
    
    return results


def run_multi_circuit_comparison(
    num_qubits_range: Tuple[int, int] = (2, 7),
    circuits_per_qubit: int = 5,  # Reduced from 50 to 5
    shots: int = 1024,
    max_depth: int = 10,
    save_results: bool = True,
    results_file: str = None
) -> Dict[str, Any]:
    """Run comprehensive fidelity comparison across multiple qubit counts.
    
    Args:
        num_qubits_range: (min_qubits, max_qubits) range to test
        circuits_per_qubit: Number of random circuits per qubit count
        shots: Number of shots per circuit execution
        max_depth: Maximum circuit depth for random generation
        save_results: Whether to save results to file
        results_file: Custom filename for results (auto-generated if None)
    
    Returns:
        Dictionary containing fidelity statistics for each backend and qubit count
    """
    print("\n=== MULTI-CIRCUIT FIDELITY COMPARISON ===\n")
    
    fake_torino = FakeTorino()
    all_results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'num_qubits_range': num_qubits_range,
            'circuits_per_qubit': circuits_per_qubit,
            'shots': shots,
            'max_depth': max_depth
        },
        'raw_data': {},  # Circuit-by-circuit results
        'statistics': {}  # Mean/std for each qubit count
    }
    
    for num_qubits in range(num_qubits_range[0], num_qubits_range[1] + 1):
        print(f"\n--- Testing {num_qubits}-qubit circuits ({circuits_per_qubit} circuits) ---")
        
        qubit_results = {
            'noiseless': [],
            'fake_torino': [],
            'heron': []
        }
        
        for i in range(circuits_per_qubit):
            # Progress tracking
            progress = i + 1
            if progress % max(1, circuits_per_qubit // 5) == 0 or progress == circuits_per_qubit:
                print(f"  Progress: {progress}/{circuits_per_qubit} circuits ({100*progress/circuits_per_qubit:.0f}%)")
            
            try:
                # Generate random circuit
                circuit_depth = min(max_depth, max(3, num_qubits * 2))
                circuit = generate_random_circuit_with_torino_gates(
                    num_qubits=num_qubits,
                    depth=circuit_depth,
                    seed=42 + num_qubits * 1000 + i
                )
                
                # Transpile for QubitTracking
                transpiled = transpile(circuit, backend=fake_torino, optimization_level=1)
                qt = QubitTracking(fake_torino, transpiled)
                heron_noise = HeronNoise.get_noise(qt, fake_torino)
                
                # Run comparison
                result = compare_circuit_execution(
                    circuit=circuit,
                    shots=shots,
                    custom_backend=fake_torino,
                    custom_noise_model=heron_noise
                )
                
                # Store fidelities (now using proper fidelity metric)
                qubit_results['noiseless'].append(result['noiseless']['fidelity'])
                qubit_results['fake_torino'].append(result['fake_torino']['fidelity']) 
                qubit_results['heron'].append(result['custom_noise']['fidelity'])
                
            except Exception as e:
                print(f"    Warning: Circuit {i+1} failed: {e}")
                continue
        
        # Calculate statistics
        stats = {}
        for backend_name, fidelities in qubit_results.items():
            if fidelities:
                stats[backend_name] = {
                    'mean': float(np.mean(fidelities)),
                    'std': float(np.std(fidelities)),
                    'min': float(np.min(fidelities)),
                    'max': float(np.max(fidelities)),
                    'count': len(fidelities)
                }
            else:
                stats[backend_name] = {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0}
        
        all_results['raw_data'][str(num_qubits)] = qubit_results
        all_results['statistics'][str(num_qubits)] = stats
        
        # Print summary for this qubit count
        print(f"  Results for {num_qubits} qubits:")
        for backend, stat in stats.items():
            if stat['count'] > 0:
                print(f"    {backend}: {stat['mean']:.3f} ± {stat['std']:.3f} (n={stat['count']})")
    
    # Save results if requested
    if save_results:
        if results_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"experiment_results/fidelity_comparison_{timestamp}.json"
        
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return all_results


def plot_fidelity_comparison(
    results: Dict[str, Any],
    save_plot: bool = True,
    plot_file: str = None,
    show_plot: bool = False
) -> None:
    """Create bar plot comparing fidelities across different qubit counts.
    
    Args:
        results: Results dictionary from run_multi_circuit_comparison
        save_plot: Whether to save plot to file
        plot_file: Custom filename for plot (auto-generated if None)
        show_plot: Whether to display plot interactively
    """
    # Extract data for plotting
    qubit_counts = sorted([int(k) for k in results['statistics'].keys()])
    backend_names = ['noiseless', 'fake_torino', 'heron']
    backend_labels = ['Noiseless', 'FakeTorino', 'Heron Model']
    colors = ['#2E8B57', '#DC143C', '#4169E1']  # Green, Red, Blue
    
    means = {backend: [] for backend in backend_names}
    stds = {backend: [] for backend in backend_names}
    
    for qubits in qubit_counts:
        stats = results['statistics'][str(qubits)]
        for backend in backend_names:
            means[backend].append(stats[backend]['mean'])
            stds[backend].append(stats[backend]['std'])
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Set bar width and positions
    bar_width = 0.25
    x_pos = np.arange(len(qubit_counts))
    
    # Create bars for each backend
    for i, (backend, label, color) in enumerate(zip(backend_names, backend_labels, colors)):
        bars = ax.bar(
            x_pos + i * bar_width,
            means[backend],
            bar_width,
            yerr=stds[backend],
            label=label,
            color=color,
            alpha=0.8,
            capsize=5
        )
    
    # Customize the plot
    ax.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Fidelity', fontsize=12, fontweight='bold')
    ax.set_title('Quantum Circuit Fidelity Comparison\n(Error bars show standard deviation)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(qubit_counts)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add some statistics as text
    metadata = results['metadata']
    info_text = f"Circuits per qubit: {metadata['circuits_per_qubit']}\nShots per circuit: {metadata['shots']}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save plot if requested
    if save_plot:
        if plot_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = f"plots/fidelity_comparison_{timestamp}.png"
        
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def load_comparison_results(results_file: str) -> Dict[str, Any]:
    """Load previously saved comparison results from file.
    
    Args:
        results_file: Path to JSON results file
        
    Returns:
        Results dictionary
    """
    with open(results_file, 'r') as f:
        return json.load(f)


def run_complete_fidelity_study(
    num_qubits_range: Tuple[int, int] = (2, 7),
    circuits_per_qubit: int = 50,
    shots: int = 1024
) -> str:
    """Run the complete fidelity study and generate plots.
    
    Args:
        num_qubits_range: Range of qubit counts to test
        circuits_per_qubit: Number of circuits per qubit count
        shots: Shots per circuit
        
    Returns:
        Path to the generated plot file
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE QUANTUM FIDELITY COMPARISON STUDY")
    print("="*60)
    
    # Run the multi-circuit comparison
    results = run_multi_circuit_comparison(
        num_qubits_range=num_qubits_range,
        circuits_per_qubit=circuits_per_qubit,
        shots=shots
    )
    
    # Generate the plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"plots/fidelity_study_{timestamp}.png"
    
    plot_fidelity_comparison(results, save_plot=True, plot_file=plot_file)
    
    # Print summary
    print("\n" + "="*60)
    print("STUDY SUMMARY")
    print("="*60)
    
    for qubits in sorted([int(k) for k in results['statistics'].keys()]):
        stats = results['statistics'][str(qubits)]
        print(f"\n{qubits}-qubit circuits:")
        for backend, stat in stats.items():
            if stat['count'] > 0:
                print(f"  {backend.replace('_', ' ').title()}: {stat['mean']:.3f} ± {stat['std']:.3f}")
    
    print(f"\nPlot saved to: {plot_file}")
    return plot_file


def quick_fidelity_test():
    """Quick test with fewer circuits for development/testing."""
    return run_complete_fidelity_study(
        num_qubits_range=(2, 4),
        circuits_per_qubit=5,
        shots=512
    )


def medium_fidelity_test():
    """Medium test with reasonable number of circuits."""
    return run_complete_fidelity_study(
        num_qubits_range=(2, 6),
        circuits_per_qubit=20,
        shots=1024
    )


# Quick usage examples:
# - Quick test: quick_fidelity_test()
# - Medium test: medium_fidelity_test()  
# - Full study: run_complete_fidelity_study()
# - Custom: run_complete_fidelity_study(num_qubits_range=(2,5), circuits_per_qubit=30)


def example_ibm_cloud_submission():
    """Example: Submit random circuits to IBM cloud."""
    print("=== IBM CLOUD SUBMISSION EXAMPLE ===")
    
    # Check available backends
    print("Checking available IBM backends...")
    backends = get_available_ibm_backends(min_qubits=3)
    if not backends:
        print("No IBM backends available. Make sure you're authenticated.")
        return
    
    # Submit a small batch for testing
    result = compare_random_circuits_with_ibm_cloud(
        num_qubits=3,
        depth=4, 
        num_circuits=3,
        shots=1024,
        seed=42,
        submit_only=True  # Only submit, don't wait for results
    )
    
    print(f"\nJob submitted with ID: {result['job_info']['job_id']}")
    print("Use this job ID to retrieve results later!")
    return result['job_info']['job_id']


def example_ibm_cloud_analysis(job_id: str):
    """Example: Analyze results from IBM cloud job."""
    print(f"=== IBM CLOUD ANALYSIS EXAMPLE ===")
    print(f"Analyzing job: {job_id}")
    
    result = compare_random_circuits_with_ibm_cloud(
        job_id=job_id,
        noise_type="heron"
    )
    
    if 'error' not in result:
        print("Analysis completed successfully!")
        return result
    else:
        print(f"Error: {result['error']}")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "ibm_submit":
            # Submit jobs to IBM cloud
            job_id = example_ibm_cloud_submission()
            
        elif mode == "ibm_analyze" and len(sys.argv) > 2:
            # Analyze existing IBM job
            job_id = sys.argv[2]
            example_ibm_cloud_analysis(job_id)
            
        elif mode == "backends":
            # List available backends
            print("=== AVAILABLE IBM BACKENDS ===")
            backends = get_available_ibm_backends()
            for backend in backends:
                print(f"{backend['name']}: {backend['num_qubits']} qubits, "
                      f"queue: {backend['pending_jobs']}, "
                      f"status: {'✓' if backend['operational'] else '✗'}")
        
        else:
            print("Usage:")
            print("  python noise_model_comparison.py ibm_submit     # Submit job to IBM")
            print("  python noise_model_comparison.py ibm_analyze <job_id>  # Analyze results")
            print("  python noise_model_comparison.py backends       # List IBM backends")
    
    else:
        # Default: Run original examples
        print("=== BELL STATE COMPARISON (Original) ===")
        example_bell_state_comparison()
        
        print("\n\n=== RANDOM CIRCUIT COMPARISON (Original) ===")
        example_random_circuit_comparison()
        
        print("\n\n" + "="*60)
        print("NEW IBM CLOUD FUNCTIONALITY:")
        print("  python noise_model_comparison.py ibm_submit     # Submit to IBM cloud")
        print("  python noise_model_comparison.py ibm_analyze <job_id>  # Analyze results")  
        print("  python noise_model_comparison.py backends       # List backends")
        print("="*60)


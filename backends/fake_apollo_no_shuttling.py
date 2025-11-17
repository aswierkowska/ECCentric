import numpy as np
from qiskit.transpiler import InstructionProperties, Target, CouplingMap
from qiskit.circuit.library import RZGate, XGate, SXGate, CZGate, Measure, Reset
# TODO: Delay
from qiskit.circuit import Parameter
from qiskit.providers import BackendV2, Options
from qiskit.visualization import plot_coupling_map

class FakeQuantinuumApolloNoShuttlingBackend(BackendV2):
    """Fake Quantinnum Apollo Backend without shuttling."""
    
    def __init__(self):
        super().__init__(name="FakeQuantinuumApolloNoShuttling", backend_version=2)
        # if 192 probably 12 x 16
        # then 1000s could be 1728 (36*48) or 3072 (48*64)
        self.rows = 24
        self.columns = 32
        self._coupling_map = CouplingMap.from_grid(self.rows, self.columns)
        self._num_qubits = self._coupling_map.size()
        self._target = Target("Fake Quantinuum Apollo No Shuttling", num_qubits=self._num_qubits)
        self._remote_gates = {}

    @property
    def target(self):
        return self._target
    
    @property
    def max_circuits(self):
        return None
    
    @property
    def coupling_map(self):
        return self._coupling_map
    
    @property
    def qubit_positions(self):
        return self._qubit_positions
    
    @property
    def get_remote_gates(self):
        return self._remote_gates
    
    @property
    def num_qubits(self):
        return self._num_qubits
    
    @property
    def coupling_map(self):
        return self._coupling_map
    
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
  
    def run(self, circuit, **kwargs):
        raise NotImplementedError("This backend does not contain a run method")

if __name__ == "__main__":
    backend = FakeQuantinuumApolloNoShuttingBackend()
    #plot_coupling_map(backend.coupling_map.size(), None, backend.coupling_map.get_edges(), filename="apollo.png")
    print(backend.get_remote_gates)

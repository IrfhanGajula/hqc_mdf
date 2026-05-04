import pennylane as qml
from hqc_mdf.config.framework_config import QUBITS, DEVICE_TYPE

# Initialize the shared quantum device
dev = qml.device(DEVICE_TYPE, wires=QUBITS)

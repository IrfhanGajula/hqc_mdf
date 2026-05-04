import pennylane as qml
from .quantum_encoder import angle_encoding
from .quantum_executor import dev

@qml.qnode(dev)
def standard_vqc(features, weights):
    """
    Standard VQC with basic entanglement.
    """
    angle_encoding(features)
    # Uses 1 layer by default if weights shape is (1, qubits, 3)
    qml.StronglyEntanglingLayers(weights, wires=range(len(features)))
    return qml.expval(qml.PauliZ(0))

@qml.qnode(dev)
def advanced_vqc(features, weights):
    """
    Advanced VQC with deeper entanglement and feature repetition.
    """
    angle_encoding(features)
    # Weights should be (Layers, Qubits, 3)
    qml.StronglyEntanglingLayers(weights, wires=range(len(features)))
    return qml.expval(qml.PauliZ(0))

import pennylane as qml

def angle_encoding(features):
    """
    Encodes classical features into quantum states using Angle Encoding.
    Args:
        features: Array of features (size should match number of qubits).
    """
    qml.AngleEmbedding(features, wires=range(len(features)), rotation='X')

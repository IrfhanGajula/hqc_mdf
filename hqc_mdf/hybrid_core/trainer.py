import pennylane as qml
from pennylane import numpy as np
from hqc_mdf.config.framework_config import QUBITS, LEARNING_RATE, EPOCHS
import os
import time

def train_model(X_train, y_train, circuit_fn, batch_size=5, layers=1):
    """
    Trains a specific VQC configuration and returns weights + technical metadata.
    """
    weights = np.random.random((layers, QUBITS, 3), requires_grad=True)
    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    
    print(f"\nTraining Quantum Model ({circuit_fn.__name__})...")
    print(f"Config: {QUBITS} qubits, {layers} layers, {EPOCHS} epochs.")
    
    history = []
    num_samples = len(X_train)
    start_time = time.time()

    for epoch in range(EPOCHS):
        indices = np.random.permutation(num_samples)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]
        
        epoch_costs = []
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            if len(X_batch) == 0: continue
                
            weights, cost = opt.step_and_cost(lambda w: cost_function(w, X_batch, y_batch, circuit_fn), weights)
            epoch_costs.append(cost)
        
        avg_cost = np.mean(epoch_costs)
        history.append(avg_cost)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1}/{EPOCHS} | Cost: {avg_cost:.6f}")
            
    end_time = time.time()
    
    # Technical Analysis
    initial_cost = history[0]
    final_cost = history[-1]
    convergence_slope = (initial_cost - final_cost) / EPOCHS
    
    training_md = {
        "initial_cost": initial_cost,
        "final_cost": final_cost,
        "convergence_slope": convergence_slope,
        "training_time": end_time - start_time,
        "layers": layers,
        "hilbert_space_dims": 2**QUBITS
    }
    
    print(f"Convergence Quality: {convergence_slope:.6f} delta/epoch")
    return weights, training_md

def cost_function(weights, features, targets, circuit_fn):
    """
    MSE Loss for the specified circuit.
    """
    mapped_targets = 2 * targets - 1
    # Note: execute_vqc needs to be updated or circuit_fn used directly
    predictions = np.array([circuit_fn(f, weights) for f in features])
    return np.mean((predictions - mapped_targets) ** 2)


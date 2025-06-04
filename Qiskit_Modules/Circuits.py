from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RXGate, RYGate, CZGate, MCXGate, RZGate, MCMT

import numpy as np

def build_rx(N, R):
    rx_circuit = QuantumCircuit(N + R + 1)

    # apply Rx rotation gates
    for i in range(N):
        for r in range(R):
            gate = RXGate(np.pi / (2**r))
            rx_circuit.append(gate.control(1), (N + 1 + r, i))

    # apply CNOT gate to target
    rx_circuit.append(MCXGate(N), list(range(0, N+1)))

    return rx_circuit

def build_ry(N, R):
    ry_circuit = QuantumCircuit(N + R + 1)

    # apply Rx rotation gates
    for i in range(N):
        for r in range(R):
            gate = RYGate(np.pi / (2**r))
            ry_circuit.append(gate.control(1), (N + 1 + r, i))

    # apply CNOT gate to target
    ry_circuit.append(MCXGate(N), list(range(0, N+1)))

    return ry_circuit

def build_transition(N):
    transition_circ = QuantumCircuit(2*N + 1)

    for i in range(N):
        transition_circ.append(MCXGate(2), (i, 2*N, N + i))

    return transition_circ

def build_transformation(N, R):
    # R0lstar circuit
    transformation_circ = QuantumCircuit(N + R + 1)

    # append X gates
    transformation_circ.x(list(range(N+1, N+R+1)))

    # apply CZ gate
    cz = RZGate(np.pi)
    indices = list(range(0, N)) + list(range(N+1, N+R+1)) + [N]
    transformation_circ.append(MCMT(cz, N+R, 1), indices)   
    # undo X gates
    transformation_circ.x(list(range(N+1, N+R+1)))

    return transformation_circ


def build_circuit(X, R):
    '''
    args:
        X: array of data points
        R: rotation precision
    returns:
        circuit: complete circuit
    '''
    # convert to NP array
    X = np.array(X)

    N = X.shape[0]
    B = X.shape[1]

    qubits = QuantumRegister((N + 2 + 2*R) * B + B - 1)
    cbits = ClassicalRegister(1)

    # initialize circuit
    circuit = QuantumCircuit(qubits, cbits)

    circuit.h(list(range(N*B + 2*B, (N + 2 + 2*R) * B + B - 1)))

    rx_block = build_rx(N, R)
    ry_block = build_ry(N, R)
    transition_block = build_transition(N)
    transformation_block = build_transformation(N, R)

    for i in range(B):
        # apply RX block (data + target + rotation)
        rx_lanes = list(range(i * N, (i + 1) * N)) + [N*B + 2*i] + list(range(N*B + 2*B + 2*R*i, N*B + 2*B + 2*R*i + R))
        circuit.append(rx_block, rx_lanes)

        for d in range(N):
            if X[d][i] == 0:
                circuit.x(...)
        
        # measure

        # OAA

        # apply Ry block
        # list(range(i * N, (i + 1) * N)) + [N*B + 2*i] + list(range(N*B + 2*B + 2*R*i, N*B + 2*B + 2*R*i + R))
        ry_lanes = list(range(i * N, (i + 1) * N)) + [N*B + 2*i + 1] + list(range(N*B + 2*B + 2*R*i + R, N*B + 2*B + 2*R*i + 2*R))
        circuit.append(ry_block, ry_lanes)
        
        for d in range(N):
            if X[d][i] == 0:
                circuit.x(...)
        
        # measure

        # OAA
        

        if i != B - 1:
            # apply transition block
            ...

    

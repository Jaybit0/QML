from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import RXGate, RYGate, CZGate, MCXGate, RZGate, MCMT

import numpy as np

# TODO: add comments

def build_transition(N):
    transition_circ = QuantumCircuit(2*N + 1)

    for i in range(N):
        transition_circ.append(MCXGate(2), (i, 2*N, N + i))

    return transition_circ

def build_r0lstar(R):
    # target + all of the R rotation controls
    r0lstar_circuit = QuantumCircuit(R)

    r0lstar_circuit.x(list(range(R)))
    r0lstar_circuit.mcrz(np.pi, list(range(R-1)), R-1)
    r0lstar_circuit.x(list(range(R)))

    return r0lstar_circuit

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

	N = X.shape[0] # number of data points
	B = X.shape[1] # number of bits

	size = (N + 2 + 2*R) * B + B - 1
	num_rotation_param = 2 * R * B

	qr = QuantumRegister(size)
	cr = ClassicalRegister(num_rotation_param, name='cr')
	xr = ClassicalRegister(B, name='xr')
	yr = ClassicalRegister(B, name='yr')

	# initialize circuit
	circuit = QuantumCircuit(qr, cr, xr, yr)

	circuit.h(list(range(N*B + 2*B, (N + 2 + 2*R) * B + B - 1)))

	transition_block = build_transition(N).to_gate(label='transition block')
	r0lstar = build_r0lstar(R).to_gate(label='r0lstar')

	for i in range(B):
		# apply RX block (data + target + rotation)
		rx_target_index = N*B + 2*i
		rx_lanes = list(range(i * N, (i + 1) * N)) + [rx_target_index] + list(range(N*B + 2*B + 2*R*i, N*B + 2*B + 2*R*i + R))

		# add the X-gates for bits with 0
		rx_circuit = QuantumCircuit(N + R + 1)

		# apply Rx rotation gates
		for j in range(N):
			for r in range(R):
				gate = RXGate(np.pi / (2**r))
				rx_circuit.append(gate.control(1), (N + 1 + r, j))
				
		# rx_bit_i = QuantumCircuit(rx_block.num_qubits)
		# rx_bit_i.append(rx_block, list(range(rx_block.num_qubits)))
		for d in range(N):
			if X[d][i] == 0:
				# add X values for all bits with 0's
				rx_circuit.x(d)

		# apply CNOT gate to target
		rx_circuit.append(MCXGate(N), list(range(0, N+1)))
		rx_circuit = rx_circuit.to_gate(label='ctrl-Rx block')

		circuit.append(rx_circuit, rx_lanes)
		
		# measure
		circuit.measure(rx_target_index, xr[i])

		# OAA
		with circuit.if_test((xr[i], 0)):
			rx_inv = rx_circuit.inverse()
			rx_inv.label = 'ctrl-Rx inverse'
			circuit.append(rx_inv, rx_lanes)
			r0lstar_lanes = list(range(N*B + 2*B + 2*R*i, N*B + 2*B + 2*R*i + R))
			circuit.append(r0lstar, r0lstar_lanes)
			circuit.append(rx_circuit, rx_lanes)

		# apply Ry block
		ry_target_index = N*B + 2*i + 1
		ry_lanes = list(range(i * N, (i + 1) * N)) + [ry_target_index] + list(range(N*B + 2*B + 2*R*i + R, N*B + 2*B + 2*R*i + 2*R))

		# add the X gates for bits with 0
		ry_circuit = QuantumCircuit(N + R + 1)

		# apply Rx rotation gates
		for j in range(N):
			for r in range(R):
				gate = RYGate(np.pi / (2**r))
				ry_circuit.append(gate.control(1), (N + 1 + r, j))
				
		# ry_bit_i = QuantumCircuit(ry_block.num_qubits)
		# ry_bit_i.append(rx_block, list(range(ry_block.num_qubits)))
		for d in range(N):
			if X[d][i] == 0:
				# add X values for all bits with 0's
				ry_circuit.x(d)

		# apply CNOT gate to target
		ry_circuit.append(MCXGate(N), list(range(0, N+1)))
		ry_circuit = ry_circuit.to_gate(label='ctrl-Ry block')

		circuit.append(ry_circuit, ry_lanes)
		
		# measure
		circuit.measure(ry_target_index, yr[i])

		# OAA
		with circuit.if_test((yr[i], 0)):
			ry_inv = ry_circuit.inverse()
			ry_inv.label = 'ctrl-Ry inverse'
			circuit.append(ry_inv, ry_lanes)
			r0lstar_lanes = list(range(N*B + 2*B + 2*R*i + R, N*B + 2*B + 2*R*i + 2*R))
			circuit.append(r0lstar, r0lstar_lanes)
			circuit.append(ry_circuit, ry_lanes)
		
		if i != B - 1:
			# apply the transition from bit i to bit i + 1
			transition_lanes = list(range(i * N, (i + 2) * N)) + [size - B + i] # [consecutive lanes] + [control]
			# apply transition block
			circuit.append(transition_block, transition_lanes)
			
	circuit.measure(qr[N*B + 2*B:N*B + 2*B + 2*R*B], cr)

	return circuit


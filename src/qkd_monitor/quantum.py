"""Exact single-qubit Qiskit kernels and independent Aer shot validation.

Intercept/resend with unobserved Eve outcome is dephasing in Eve's basis.
Depolarizing channel has Pauli probabilities (1-3q/2,q/2,q/2,q/2),
so its matched-basis QBER is q. It is affine in q on [0,.5].
"""
import itertools
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import DensityMatrix, Kraus
from qiskit_aer import AerSimulator
from qiskit_aer.noise import pauli_error

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.diag([1, -1]).astype(complex)


def circuit(bit, alice_basis, bob_basis, eve_basis, attacked, q, measure=False):
    qc = QuantumCircuit(1, 1) if measure else QuantumCircuit(1)
    if bit:
        qc.x(0)
    if alice_basis:
        qc.h(0)
    if attacked:
        if eve_basis:
            qc.h(0)
        qc.append(pauli_error([("I", .5), ("Z", .5)]).to_instruction(), [0])
        if eve_basis:
            qc.h(0)
    qc.append(pauli_error([("I", 1 - 1.5*q), ("X", .5*q),
                                 ("Y", .5*q), ("Z", .5*q)]).to_instruction(), [0])
    if bob_basis:
        qc.h(0)
    if measure:
        qc.measure(0, 0)
    return qc


def exact_p1(bit, alice_basis, bob_basis, eve_basis, attacked, q):
    qc = QuantumCircuit(1)
    if bit:
        qc.x(0)
    if alice_basis:
        qc.h(0)
    rho = DensityMatrix.from_instruction(qc)
    if attacked:
        rotate = QuantumCircuit(1)
        if eve_basis:
            rotate.h(0)
        rho = rho.evolve(rotate).evolve(Kraus([I/np.sqrt(2), Z/np.sqrt(2)])).evolve(rotate)
    rho = rho.evolve(Kraus([np.sqrt(1-1.5*q)*I, np.sqrt(.5*q)*X,
                           np.sqrt(.5*q)*Y, np.sqrt(.5*q)*Z]))
    read = QuantumCircuit(1)
    if bob_basis:
        read.h(0)
    return float(rho.evolve(read).probabilities()[1])


class QiskitKernel:
    def __init__(self):
        self.endpoints = np.empty((2, 2, 2, 2, 2, 2))
        for bit, a, b, e, attack in itertools.product(range(2), repeat=5):
            for endpoint, q in enumerate((0., .5)):
                self.endpoints[bit, a, b, e, attack, endpoint] = exact_p1(bit, a, b, e, attack, q)

    def probabilities(self, bits, a, b, e, attack, q):
        if np.any((q < 0) | (q > .5)):
            raise ValueError("Channel QBER must lie in [0,.5]")
        endpoints = self.endpoints[bits, a, b, e, attack.astype(int)]
        return np.clip(endpoints[:, 0] + 2*q*(endpoints[:, 1]-endpoints[:, 0]), 0, 1)


def validate_aer(shots=16000, seed=20261011):
    """Shot comparison of all 32 settings at three channel strengths."""
    settings = [(*s, q) for q in (0., .02, .08)
                for s in itertools.product(range(2), repeat=5)]
    backend = AerSimulator(method="density_matrix", max_parallel_threads=2)
    circuits = [circuit(*s, measure=True) for s in settings]
    compiled = transpile(circuits, backend, optimization_level=0, seed_transpiler=seed)
    result = backend.run(compiled, shots=shots, seed_simulator=seed).result()
    rows = []
    for i, s in enumerate(settings):
        exact = exact_p1(*s)
        empirical = result.get_counts(i).get("1", 0)/shots
        # Hoeffding union bound over all settings, family error <= 1e-6.
        tolerance = np.sqrt(np.log(2*len(settings)/1e-6)/(2*shots))
        rows.append(dict(bit=s[0], alice_basis=s[1], bob_basis=s[2], eve_basis=s[3],
                         attacked=s[4], noise_q=s[5], exact_p1=exact, aer_p1=empirical,
                         abs_error=abs(exact-empirical), tolerance=tolerance,
                         passed=bool(abs(exact-empirical) <= tolerance), shots=shots))
    return rows

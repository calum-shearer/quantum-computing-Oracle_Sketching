from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import DiagonalGate, UnitaryGate
from qiskit.quantum_info import Operator 
from numpy.typing import ArrayLike, NDArray
import numpy as np
import pennylane as qp
import math
import pyqsp
from pyqsp import angle_sequence
import matplotlib.pyplot as plt

# Adapted from https://pennylane.ai/demos/tutorial_apply_qsvt to work with Sign function
def get_sign_phases(degree:int=41,delta:int=15,max_scale:float=0.99) -> tuple[list[float], float]:
    ''' Returns phases of the sign operator approximated to specified degree,
    using erf(delta * x). Phases in format suitable to apply pennyline qp.QSVT
    after converting using qp.PCPhase, or with custom qiskitQSVTgate()'''
    pcoefs, scale = pyqsp.poly.PolySign().generate(degree=degree,
                delta=delta,
                ensure_bounded=True,
                return_scale=True,
                chebyshev_basis=False,
                cheb_samples=20,
                max_scale=max_scale)
    phi_pyqsp = pyqsp.angle_sequence.QuantumSignalProcessingPhases(pcoefs, signal_operator="Wx", tolerance=0.001,chebyshev_basis=False) #method not sym_qsp
    phi_qsvt = qp.transform_angles(phi_pyqsp, "QSP", "QSVT")
    return phi_qsvt, scale

# from https://pennylane.ai/demos/tutorial_apply_qsvt
def check_sign_approx(phases: list[float], scale: float) -> None:
    x_vals = np.linspace(-1, 1, 50)
    target_y_vals = [scale * np.sign(x) for x in np.linspace(-1, 1, 50)]

    qsvt_y_vals = []
    for x in x_vals:

        block_encoding = qp.BlockEncode(x, wires=[0])
        projectors = [qp.PCPhase(angle, dim=1, wires=[0]) for angle in phases]

        poly_x = qp.matrix(qp.QSVT, wire_order=[0])(block_encoding, projectors)
        qsvt_y_vals.append(np.real(poly_x[0][0]))
        
    plt.plot(x_vals, np.array(qsvt_y_vals), label="Re(qsvt)")
    plt.plot(np.linspace(-1, 1, 50), target_y_vals, label="target")

    plt.vlines(0.0, -1.0, 1.0, color="black")
    plt.hlines(0.0, -0.1, 1.0, color="black")

    plt.legend()
    plt.show()

def qiskitPCPhase(phi: float, dim: int, num_qubits: int) -> DiagonalGate:
    '''Pennylane PCPhase for qiskit: see
    https://docs.pennylane.ai/en/stable/code/api/pennylane.PCPhase.html'''
    qc = QuantumCircuit(num_qubits)
    signs = np.concatenate((np.ones(dim), -1*np.ones(2**num_qubits - dim)))
    phase = signs * phi
    PCPhasegate = DiagonalGate(np.exp(1j*phase))
    PCPhasegate.label= r'$\Pi_{\phi}$'
    return PCPhasegate

def qiskitMatrix_to_QSVTgate(matrix:ArrayLike, phases: list[float] ,encoded=False) -> UnitaryGate:
    """Quantum singular value transformation circuit for qiskit,
    given Hermitian A or its block encoding Unitary and specified list of phases.
    Requires pennylane import if not alread block encoded"""
    numqubits =  math.ceil(math.log2(len(matrix)))   
    if encoded == False:
        block_encoding = qp.matrix(qp.BlockEncode(matrix,wires=range(numqubits+1)))
        numqubits += 1
    else:
        block_encoding = matrix

    Encoder_gate = UnitaryGate(block_encoding,label=r'$U$')
    adjoint_gate = Encoder_gate.inverse()
    adjoint_gate.label = r'$U^{\dagger}$'

    quantum_reg = QuantumRegister(numqubits-1)
    ancilla_reg = AncillaRegister(1)

    qsvt_circuit = QuantumCircuit(quantum_reg,ancilla_reg)

    iterations = int(len(phases))

    qsvt_circuit.append(qiskitPCPhase(phases[0], 1, 1),ancilla_reg)
    op = Operator(qsvt_circuit)

    for k in range(1,iterations):
        if k % 2 == 0:
                qsvt_circuit.append(Encoder_gate,range(numqubits))
                qsvt_circuit.append(qiskitPCPhase(phases[k], 1, 1),ancilla_reg)
        else:
                qsvt_circuit.append(adjoint_gate,range(numqubits))
                qsvt_circuit.append(qiskitPCPhase(phases[k], 1, 1),ancilla_reg)
    return qsvt_circuit.to_gate()

def qiskitCircuit_to_QSVTCircuit(circuit:QuantumCircuit, phases:list[float], ancilla_index:int = None) -> QuantumCircuit:
    """Quantum singular value transformation circuit for qiskit,
    given a circuit encoding a block encoded Unitary and specified list of phases.
    Polynomial from which phases were obtained must be strictly odd"""

    qubitcount = circuit.num_qubits

    if ancilla_index == None:
        ancilla_index = qubitcount-1

    encoder_circuit = circuit.to_gate(label=circuit.name)
    
    adjoint_circuit = encoder_circuit.inverse()

    iterations = int(len(phases))

    QSVT_circuit = QuantumCircuit(qubitcount)
    QSVT_circuit.append(qiskitPCPhase(phases[0], 1, 1),qargs = [ancilla_index])

    for k in range(1,iterations):
        if k % 2 == 0:
                QSVT_circuit.append(encoder_circuit,range(qubitcount))
                QSVT_circuit.append(qiskitPCPhase(phases[k], 1, 1),qargs = [ancilla_index])
        else:
                QSVT_circuit.append(adjoint_circuit,range(qubitcount))
                QSVT_circuit.append(qiskitPCPhase(phases[k], 1, 1),qargs = [ancilla_index])

    return QSVT_circuit

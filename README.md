In this reposIn this notebook we implement toy examples for the Oracle Sketching procedure from appendix D of the paper "Exponential quantum advantage in processing massive classical data" by Haimeng Zhao, Alexander Zlokapa, Hartmut Neven, Ryan Babbush, John Preskill, Jarrod R. McClean, Hsin-Yuan Huang (https://arxiv.org/abs/2604.07639). We build qiskit functions that produce objects of the `QuantumCircuit` class for the following:

1. Oracle Sketching for IID data of a Boolean function (Appendix D.2)
1. Oracle Sketching for IID data of a multi-bit output function (Appendix D.4.a)
1. Element Oracle Sketching for sparse matrices (Appendix D.5.b)
1. Row and Column Index Oracle Sketching for dense matrices (Appendix D.5.b)

The file `OracleSketching.ipynb` contains code for the above examples, along with qiskit circuit diagrams and tests to verify the correctness of the circuits.

The file `oracle_functions.py` contains the majority of the functions used in the above notebook. The file `sign_function_qsvt.py` contains functions that are used when we apply the Quantum Singular Value Transformation (QSVT) when constructing the Row and Index Column Oracles. The corresponding notebook `sign_function_qsvt.ipynb` contains demonstrations and verification of these functions. Note that the code for computing phase angles and visualization for QSVT is in part adapated from https://pennylane.ai/demos/tutorial_apply_qsvt , while the code for the QSVT quantum circuit in qiskit is adapter from Pennylane's qp.QSVT function (https://docs.pennylane.ai/en/stable/code/api/pennylane.QSVT.html).

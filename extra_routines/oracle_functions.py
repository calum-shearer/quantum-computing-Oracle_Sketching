import qiskit.circuit
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import DiagonalGate, UnitaryGate, StatePreparation, IntegerComparator
import numpy as np
import math
import pyqsp
from extra_routines.sign_function_qsvt import *
from qiskit import transpile
import qiskit.quantum_info as qi
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from IPython.display import display
from numpy.typing import ArrayLike, NDArray
backend = AerSimulator()

# Function oracles
def boolean_oracle_sketching_circuit(domain_size, sample_data, time):
    ''' Implements Algorithm 1: Quantum Oracle sketching as described above. Outputs appropriate QuantumCircuit object'''

    num_qubits = int(math.ceil(math.log2(domain_size)))
    quantum_register = QuantumRegister(num_qubits,name='q')

    oracle_circuit = QuantumCircuit(quantum_register)   
    
    sample_size = len(sample_data)

    for i in range(sample_size):
        if int(sample_data[i][1]) == 1:
            target_state = int(sample_data[i][0])
            target_string = f'{target_state:0{num_qubits}b}'[::-1] #Reversed for Little Endian
            oracle_circuit.barrier()
            for k in range(num_qubits):
                if target_string[k] == '0':
                    oracle_circuit.x(quantum_register[k])
            oracle_circuit.barrier()
            oracle_circuit.mcp(time/sample_size, control_qubits = quantum_register[1:], target_qubit = quantum_register[0])
            oracle_circuit.barrier()
            for k in range(num_qubits):
                if target_string[k] == '0':
                    oracle_circuit.x(quantum_register[k])
    return oracle_circuit

def multi_oracle_sketching_circuit(domain_size, codomain_size, sample_data, time):
    ''' Implements Algorithm 1: Quantum Oracle sketching as described above. Outputs appropriate QuantumCircuit object. 
    Underlying function from [domain_size] -> [codomain_size].'''

    bit_length = int(math.ceil(math.log2(codomain_size)))

    num_qubits_x = int(math.ceil(math.log2(domain_size)))
    num_qubits_b = int(math.ceil(math.log2(bit_length)))

    quantum_register_x = QuantumRegister(num_qubits_x,name='x')
    quantum_register_b = QuantumRegister(num_qubits_b,name='b')

    oracle_circuit = QuantumCircuit(quantum_register_x,quantum_register_b)   
    
    sample_size = len(sample_data)

    for i in range(sample_size):
        if int(sample_data[i][1]) != 0:
            
            target_state_x = int(sample_data[i][0])
            target_string_x = f'{target_state_x:0{num_qubits_x}b}'[::-1]
    
            target_state_b = int(sample_data[i][1])
            target_string_b = f'{target_state_b:0{bit_length}b}'[::-1]
            print(f'f({target_state_x}) = {target_state_b}')
            print('binary=',target_string_x,target_string_b)

            for m in range(num_qubits_x): # send x_i to all 1s
                if int(target_string_x[m]) == 0:
                        oracle_circuit.x(quantum_register_x[m])

            oracle_circuit.barrier()

            for k in range(bit_length): # iterate over all possible states the b_register can be in
                y_i_j = int(target_string_b[k]) #tells us phase
                k_string = f'{k:0{num_qubits_b}b}'[::-1]
                # print('k=',k,'k_string=',k_string)
                for l in range(num_qubits_b): # swap k to all 1s
                    if int(k_string[l]) == 0:
                        oracle_circuit.x(quantum_register_b[l])

                oracle_circuit.mcp(time*y_i_j/sample_size, control_qubits = quantum_register_x[:] + quantum_register_b[:-1], target_qubit = quantum_register_b[-1])

                for l in range(num_qubits_b): # swap 1 back to k
                    if int(k_string[l]) == 0:
                        oracle_circuit.x(quantum_register_b[l])

                oracle_circuit.barrier()

            for m in range(num_qubits_x): # swap 1 back to x_i
                if int(target_string_x[m]) == 0:
                        oracle_circuit.x(quantum_register_x[m])
        oracle_circuit.barrier()
        oracle_circuit.barrier()

    return oracle_circuit

# Matrix Sketching Oracles
def element_oracle_gate(matrix: ArrayLike, samples: int, startingseed:int=None, samplerelative:bool=False, prints:bool=False) -> QuantumCircuit:
    ''' Element oracle for given sparse matrix with specified sample count.
    Sample count will be (number of nonzero entries)*(samples) if samplerelative=True'''

    if startingseed == None:
        startingseed = np.random.randint(1e8)

    num_qubits = required_qubits(matrix)
    element_bits = math.ceil(np.log2(np.max(matrix)))

    qr_row = QuantumRegister(num_qubits,name='q_row')
    qr_col = QuantumRegister(num_qubits,name='q_col')
    qr_entry = QuantumRegister(element_bits,name='q_entry')
    element_oracle_circuit = QuantumCircuit(qr_row,qr_col,qr_entry)

    K = np.count_nonzero(matrix) #size of sample space

    seednumber = startingseed

    sample_number=  samples

    if samplerelative==True:
        sample_number = sample_number*K

    #initialize |0> to |-> so that controlled phase marking yields |-> when undoing.
    element_oracle_circuit.x(qr_entry)
    element_oracle_circuit.h(qr_entry) 

    for m in range(sample_number):
        rsample = sparse_element_stream(matrix,seednum=seednumber) 
        seednumber += 1
        if prints == True:
            display(f'i = {rsample[0]} j = {rsample[1]} Aij = {rsample[2]}')
        for k in range(element_bits):

            row_bitstring = f'{rsample[0]:0{num_qubits}b}'
            col_bitstring = f'{rsample[1]:0{num_qubits}b}'
            element_bitstring = f'{rsample[2]:0{element_bits}b}'[::-1]

            index_bitstring = col_bitstring+row_bitstring #mcp control state reads right to left (big endian)
            #Apply random unitary V_a(i,j) for A_{i,j} bitwise on entry register

            bitint = int(element_bitstring[k])
            element_oracle_circuit.mcp(np.pi*K*bitint/sample_number,control_qubits=qr_row[:]+qr_col[:],target_qubit=qr_entry[k],ctrl_state= index_bitstring)

    element_oracle_circuit.h(qr_entry)
    element_oracle_circuit.x(qr_entry)

    return element_oracle_circuit

def rcumulative_count_unitary(matrix: ArrayLike, samplenum:int=25, startingseed:int=None, samplerelative:bool=False, row_sparsity:int=None) -> QuantumCircuit:
    ''' Implements row Cumulative Count Unitary for given matrix'''
    
    if row_sparsity is None:
        row_sparsity = np.max(np.count_nonzero(matrix, axis=1))
    
    num_qubits = math.ceil(np.log2(len(matrix)))
    sparse_bits = math.ceil(np.log2(row_sparsity+1))

    qr_row = QuantumRegister(num_qubits,name='q_i')
    qr_sparse = QuantumRegister(sparse_bits,name='q_k')
    qr_l = QuantumRegister(num_qubits,name='q_l')
    ar = AncillaRegister(1,name='a') #For the Comparator
    wr = AncillaRegister(num_qubits-1,name='w') #For the Comparator working register (comparing to qr_sparse, so need same number of ancillas to compare)

    rcumulative_unitary_circuit = QuantumCircuit(qr_row,qr_sparse,qr_l,ar,wr)

    K = np.count_nonzero(matrix) #size of sample space

    if samplerelative == True:
        sample_number = int(K*samplenum)
    else:
        sample_number = samplenum

    phasedenom = 2*(row_sparsity+0.5) # constant factor used in all phase gates
    phase_shift = np.pi*K/(sample_number*phasedenom) #constant factor used in sampled mcp gates
    
    if startingseed == None:
        startingseed = np.random.randint(1e8)

    seednumber = startingseed

    for m in range(sample_number):
        rsample = sparse_element_stream(matrix,seednum=seednumber) 
        seednumber += 1
        
        comparator_value = rsample[1]

        if comparator_value != 2**num_qubits-1: #otherwise impossible to have l > j
            comparator_circuit = IntegerComparator(num_state_qubits=num_qubits,value=comparator_value+1,geq=True).to_gate() #ancilla stores l > j
            comparator_circuit.name = rf'$l > {rsample[1]}$'
            comparator_inverse = comparator_circuit.inverse()
            comparator_inverse.name = rf'$l \leq {rsample[1]}^{{\dagger}}$'
            
            rcumulative_unitary_circuit.compose(comparator_circuit,qr_l[:]+ar[:]+wr[:],inplace=True) 
            rcumulative_unitary_circuit.mcp(phase_shift,control_qubits=qr_row[:],target_qubit=ar[0],ctrl_state= rsample[0]) #Uses ancilla to implement phase shift on all appropriate |a>
            rcumulative_unitary_circuit.compose(comparator_inverse,qr_l[:]+ar[:]+wr[:],inplace=True) # clean all ancillas!

    #Now implement offset gate U_o to get U_c
    offsetfactor = -np.pi/phasedenom 

    rcumulative_unitary_circuit.x(ar)
    rcumulative_unitary_circuit.p(offsetfactor*-0.5,ar[0]) #-1/2 part of offset gate
    rcumulative_unitary_circuit.x(ar)

    for b in range(sparse_bits):
        rcumulative_unitary_circuit.p(offsetfactor*2**b,qr_sparse[b]) # k part of offset gate

    return rcumulative_unitary_circuit

def rcumulative_unitary_encoding(matrix: ArrayLike, samplenum:int=25, startingseed:int=None, samplerelative:bool=False, row_sparsity:int=None) -> QuantumCircuit:
    '''Takes U_c to W_c for specified matrix'''
    
    if row_sparsity is None:
        row_sparsity = np.max(np.count_nonzero(matrix, axis=1))
    
    num_qubits = math.ceil(np.log2(len(matrix)))
    sparse_bits = math.ceil(np.log2(row_sparsity+1))

    qr_row = QuantumRegister(num_qubits,name='q_i')
    qr_sparse = QuantumRegister(sparse_bits,name='q_k')
    qr_l = QuantumRegister(num_qubits,name='q_l')
    encoding_ancilla = AncillaRegister(1,name='a') #Ancilla for encoding
    extra_ancillas = AncillaRegister(num_qubits,name='w') #Ancillas for U_c

    encoder_circuit = QuantumCircuit(qr_row,qr_sparse,qr_l,encoding_ancilla,extra_ancillas)

    cumulative_unitary_gate = rcumulative_count_unitary(matrix, samplenum, startingseed, samplerelative, row_sparsity)
    cumulative_unitary_gate.name = r'$U_c$'
    controlled_UC = cumulative_unitary_gate.control()

    cumulative_inverse = cumulative_unitary_gate.inverse()
    controlled_adj = cumulative_inverse.control()

    encoder_circuit.h(encoding_ancilla)
    encoder_circuit.compose(controlled_UC,encoding_ancilla[:]+qr_row[:]+qr_sparse[:]+qr_l[:]+extra_ancillas[:],inplace=True)
    encoder_circuit.x(encoding_ancilla)

    encoder_circuit.compose(controlled_adj,encoding_ancilla[:]+qr_row[:]+qr_sparse[:]+qr_l[:]+extra_ancillas[:],inplace=True)
    encoder_circuit.h(encoding_ancilla)
    
    encoder_circuit.p(3*np.pi/2,encoding_ancilla) # S = X P(3pi/2) X so omitted an X from the W_c formula
    encoder_circuit.x(encoding_ancilla)

    return encoder_circuit

def rcount_qsvt(matrix: ArrayLike, samplenum:int, startingseed:int=None,samplerelative:bool=False, row_sparsity:int=None, ancilla_index:int=None, degree:int=31, delta:int=11, max_scale:float=0.98) -> QuantumCircuit:
    ''' Applies sign function QSVT to block encoding of Cumulative Count Unitary
    corresponding to input matrix'''
    if row_sparsity is None:
        row_sparsity = np.max(np.count_nonzero(matrix, axis=1))

    if ancilla_index is None:
        ancilla_index = block_ancilla_index(matrix, row_sparsity)

    qsvt_phases, scale = get_sign_phases(degree=degree,delta=delta,max_scale=max_scale)
    readjust_angle = np.arccos(scale) #shift answers by this global scale 

    encoding_circuit = rcumulative_unitary_encoding(matrix, samplenum=samplenum, startingseed=startingseed, samplerelative=samplerelative, row_sparsity=row_sparsity)
    encoding_circuit.name = r'$W_c$'
    
    qsvt_circuit = qiskitCircuit_to_QSVTCircuit(encoding_circuit,phases=qsvt_phases, ancilla_index=ancilla_index)
    return qsvt_circuit

def rcount_phase_oracle(matrix: ArrayLike, samplenum:int=100, startingseed:int=None,samplerelative:bool=False, row_sparsity:int=None, ancilla_index:int=None, degree:int=15, delta:int=6, max_scale:float=0.98) -> QuantumCircuit:
    ''' sends |i> |k> |l> |0> to (-1)^{1_{C(i.k)<l}} |0> 
    Real part controlled by next ancilla after that
    total ancillas = n + 2 for 2^n dim matrix'''

    if row_sparsity is None:
        row_sparsity = rowsparsity(matrix)

    encoding_ancilla_index = block_ancilla_index(matrix, row_sparsity)
    
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)

    qsvt_circuit = rcount_qsvt(matrix=matrix,samplenum=samplenum,degree=degree, delta=delta, max_scale=max_scale, startingseed=startingseed,samplerelative=samplerelative,row_sparsity=row_sparsity,ancilla_index=ancilla_index)
    qsvt_adj = qsvt_circuit.inverse()
    qsvt_adj.label = r'$P_{QSVT}(W_c)^{\dagger}$'

    qsvt_control = qsvt_circuit.control(ctrl_state=0)
    qsvtadj_control = qsvt_adj.control(ctrl_state=1)

    qr_row = QuantumRegister(num_qubits,name='q_i')
    qr_sparse = QuantumRegister(sparse_bits,name='q_k')
    qr_l = QuantumRegister(num_qubits,name='q_l')
    ar = AncillaRegister(1,name='a') #For encoding
    real_ancilla = AncillaRegister(1,name='r') #To get real part of QSVT function
    wr = AncillaRegister(num_qubits,name='w') #For the Comparator and its working register

    phase_oracle_circuit = QuantumCircuit(qr_row,qr_sparse,qr_l,ar,real_ancilla,wr)

    phase_oracle_circuit.h(real_ancilla)

    phase_oracle_circuit.compose(qsvt_control,real_ancilla[:]+qr_row[:]+qr_sparse[:]+qr_l[:]+ar[:]+wr[:],inplace=True)
    phase_oracle_circuit.compose(qsvtadj_control,real_ancilla[:]+qr_row[:]+qr_sparse[:]+qr_l[:]+ar[:]+wr[:], inplace=True)

    phase_oracle_circuit.h(real_ancilla)
    
    return phase_oracle_circuit

def rcumulative_count_oracle(matrix: ArrayLike, samplenum:int=100, degree:int=17, delta:int=6, max_scale:float=0.95, startingseed:int=None,samplerelative:bool=False, row_sparsity:int=None) -> QuantumCircuit:
    '''Generates circuit O_c for specified matrix'''

    if startingseed == None:
        startingseed = np.random.randint(1e8)
    
    if row_sparsity == None:
        row_sparsity = rowsparsity(matrix)
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)

    qr_row = QuantumRegister(num_qubits,name='q_i')
    qr_sparse = QuantumRegister(sparse_bits,name='q_k')
    qr_l = QuantumRegister(num_qubits,name='q_l')
    control_ancilla = AncillaRegister(1,name='o') #ancilla for controlled V_c, encodes 1[C(i,l)<k]
    encoding_ancilla = AncillaRegister(1,name='a') #Ancilla for block encoding
    extra_ancillas = AncillaRegister(num_qubits+1,name='w') #Ancillas for U_c and Re(QSVT)

    phase_oracle_gate = rcount_phase_oracle(matrix=matrix, samplenum=samplenum, startingseed=startingseed, samplerelative=samplerelative,
    row_sparsity=row_sparsity, degree=degree, delta=delta, max_scale=max_scale)
    phase_oracle_gate.name = r'$V_c$'

    controlled_VC = phase_oracle_gate.control(1)
    controlled_VC.label = r'$cV_c$'

    cumulative_count_circuit = QuantumCircuit(qr_row,qr_sparse,qr_l,control_ancilla,encoding_ancilla,extra_ancillas)
    cumulative_count_circuit.x(control_ancilla)
    cumulative_count_circuit.h(control_ancilla)
    cumulative_count_circuit.compose(controlled_VC,control_ancilla[:]+qr_row[:]+qr_sparse[:]+qr_l[:]+encoding_ancilla[:]+extra_ancillas[:],inplace=True)
    cumulative_count_circuit.h(control_ancilla)
    cumulative_count_circuit.x(control_ancilla)

    return cumulative_count_circuit

def row_index_oracle(matrix: ArrayLike, samplenum:int=100, degree:int=17, delta:int=6, max_scale:float=0.95, startingseed:int=None,samplerelative:bool=False, row_sparsity:int=None, ancilla_index:int=None) -> QuantumCircuit:
    ''' sends |i> |k> |0> |0> to |i> |k> |j(i,k)> |0>
    '''
    if startingseed == None:
        startingseed = np.random.randint(1e8)

    if row_sparsity == None:
        row_sparsity = rowsparsity(matrix)
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)

    count_oracle_circuit = rcumulative_count_oracle(matrix,samplenum=samplenum,startingseed=startingseed,samplerelative=samplerelative,
     row_sparsity=row_sparsity, degree = degree, delta = delta,max_scale = max_scale)

    q_row = QuantumRegister(num_qubits,name='q_i')
    q_sparse = QuantumRegister(sparse_bits,name='q_k')
    q_l = QuantumRegister(num_qubits,name='q_l')
    control_ancilla = AncillaRegister(1,name='o') #Ancilla for controlled V_c, encodes 1[C(i,l)<k]
    extra_ancillas = AncillaRegister(num_qubits+2,name='w') #Ancillas for U_c

    index_oracle_circ = QuantumCircuit(q_row,q_sparse,q_l,control_ancilla,extra_ancillas)

    for m in range(num_qubits):
        bit_index = num_qubits - m - 1 #start with most significant bit for binary search
        index_oracle_circ.x(q_l[bit_index])
        index_oracle_circ.compose(count_oracle_circuit,inplace=True)
        index_oracle_circ.x(q_l[bit_index])
        index_oracle_circ.swap(q_l[bit_index],control_ancilla[0])
        index_oracle_circ.barrier()

    # Clean k register
    # Iterate over bits of k, swap with l
    index_oracle_circ.x(control_ancilla[0])
    index_oracle_circ.append(subtract_one_circuit(sparse_bits),q_sparse[:]) # k -> k-1 

    # The following leaves us with |i> |0> |j(i,k)> |0>_o
    for m in range(sparse_bits):
        index_oracle_circ.swap(q_sparse[m],control_ancilla[0])
        index_oracle_circ.compose(count_oracle_circuit,inplace=True)
        index_oracle_circ.x(q_sparse[m])
    index_oracle_circ.x(control_ancilla[0]) 


    #swap qubits in l register past k register so that we have |i> |j(i,k)> |0> |0>_o
    for k in range(sparse_bits):
        for l in range(num_qubits):
            index_oracle_circ.swap([num_qubits+(sparse_bits-k-1)+l],[num_qubits+(sparse_bits-k-1)+l+1])

    return index_oracle_circ

def row_index_oracle_quicknoclean(matrix: ArrayLike, sample_list: list[list[float]], degree:int=17, delta:int=6, max_scale:float=0.95, row_sparsity:int=None) -> QuantumCircuit:
    ''' sends |i> |k> |0> |0> to |i> |k> |j(i,k)> |0>
    '''
    if row_sparsity is None:
        row_sparsity = rowsparsity(matrix)
    sparsebits = required_sparse_bits(matrix)
    num_qubits = required_qubits(matrix)

    count_oracle_circuit = rcumulative_count_oracle_quick(matrix,sample_list=sample_list, 
    degree=degree, delta=delta, max_scale=max_scale, row_sparsity=row_sparsity)

    q_row = QuantumRegister(num_qubits,name='q_i')
    q_sparse = QuantumRegister(sparsebits,name='q_k')
    q_l = QuantumRegister(num_qubits,name='q_l')
    control_ancilla = AncillaRegister(1,name='o') #Ancilla for controlled V_c, encodes 1[C(i,l)<k]
    extra_ancillas = AncillaRegister(2,name='a') #Ancillas for U_c

    index_oracle_circ = QuantumCircuit(q_row,q_sparse,q_l,control_ancilla,extra_ancillas)

    for m in range(num_qubits):
        bit_index = num_qubits - m - 1 #start with most significant bit for binary search
        index_oracle_circ.x(q_l[bit_index])
        index_oracle_circ.compose(count_oracle_circuit,inplace=True)
        index_oracle_circ.x(q_l[bit_index])
        index_oracle_circ.swap(q_l[bit_index],control_ancilla[0])
        index_oracle_circ.barrier()

    return index_oracle_circ

# Faster sketching oracles: builds cumulative count unitary with prepared sample list and computes single diagonal phase gate to apply
def quick_rcumulative_count_unitary(matrix: ArrayLike, sample_list: list[list[int]], row_sparsity:int=None) -> QuantumCircuit:
    ''' Computes row Cumulative Count Unitary for given matrix and sample list
    as a diagonal gate to bypass integer comparator'''

    if row_sparsity == None:
        row_sparsity = rowsparsity(matrix)

    K = np.count_nonzero(matrix)

    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)
    
    total_bits = 2*num_qubits + sparse_bits
   
    phasedenom = 2*(row_sparsity+0.5) # constant factor used in all phase gates

    phase_shift = np.pi*K/(len(sample_list)*phasedenom) #constant factor K pi/M 2(s_r+1/2) used in sampled gates

    phase_list = 0.5*np.pi/phasedenom * np.ones(2**total_bits) # non-k part of U_o

    for k in range(1,row_sparsity+1): # k starts at 1
        k_val_index = k*2**(num_qubits) # translates k to its index in statevector/ matrix: past the i register

        for i in range(2**num_qubits):
            for l in range(2**num_qubits):
                phase_list[i + k_val_index + l*2**(num_qubits+sparse_bits)] += -np.pi*k/phasedenom # k part of U_o applied to |i,k,l>
        
        for m in range(len(sample_list)):
            i_index = sample_list[m][0]
            j_index = sample_list[m][1]

            for l in range(j_index + 1,2**num_qubits): # l > j
                l_val_index = l*2**(num_qubits + sparse_bits) # l is after the i and k registers
                phase_list[l_val_index + k_val_index + i_index] += phase_shift # apply to all such l and any k

    diag_entries = np.exp(1j*phase_list)
    phase_diag = DiagonalGate(diag_entries)
    phase_diag.label = r'$U_c$'

    return phase_diag

def rcumulative_encoding_quick(matrix: ArrayLike, sample_list: list[list[int]], row_sparsity:int=None) -> QuantumCircuit:
    '''Takes U_c to W_c for specified matrix. Accepts pre-prepared list of samples'''
    
    if row_sparsity is None:
        row_sparsity = np.max(np.count_nonzero(matrix, axis=1))
    
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)

    qr_row = QuantumRegister(num_qubits,name='q_i')
    qr_sparse = QuantumRegister(sparse_bits,name='q_k')
    qr_l = QuantumRegister(num_qubits,name='q_l')
    encoding_ancilla = AncillaRegister(1,name='a') #Ancilla for encoding
    
    encoder_circuit = QuantumCircuit(qr_row,qr_sparse,qr_l,encoding_ancilla)

    cumulative_unitary_gate = quick_rcumulative_count_unitary(matrix, sample_list, row_sparsity=None)
    cumulative_unitary_gate.name = r'$U_c$'
    controlled_UC = cumulative_unitary_gate.control(1)
    controlled_adj = cumulative_unitary_gate.inverse().control(1)
    
    encoder_circuit.h(encoding_ancilla)
    encoder_circuit.compose(controlled_UC,encoding_ancilla[:]+qr_row[:]+qr_sparse[:]+qr_l[:],inplace=True)
    encoder_circuit.x(encoding_ancilla)

    encoder_circuit.compose(controlled_adj,encoding_ancilla[:]+qr_row[:]+qr_sparse[:]+qr_l[:],inplace=True)
    encoder_circuit.h(encoding_ancilla)
    
    encoder_circuit.p(3*np.pi/2,encoding_ancilla) # S = X P(3pi/2) X so omitted an X from the W_c formula
    encoder_circuit.x(encoding_ancilla)
    
    return encoder_circuit

def rcount_qsvt_quick(matrix: ArrayLike, sample_list: list[list[int]], degree:int=31, delta:int=11, max_scale:float=0.98, row_sparsity:int=None) -> QuantumCircuit:
    ''' Applies sign function QSVT to block encoding of Cumulative Count Unitary
    corresponding to input matrix, provided list of samples'''
    if row_sparsity is None:
        row_sparsity = np.max(np.count_nonzero(matrix, axis=1))

    qsvt_phases, scale = get_sign_phases(degree=degree,delta=delta,max_scale=max_scale)
    readjust_angle = np.arccos(scale) #shift answers by this global scale 

    encoding_circuit = rcumulative_encoding_quick(matrix, sample_list = sample_list, row_sparsity=row_sparsity)
    encoding_circuit.name = r'$W_c$'
    
    qsvt_circuit = qiskitCircuit_to_QSVTCircuit(encoding_circuit,phases=qsvt_phases)
    return qsvt_circuit

def rcount_phase_oracle_quick(matrix: ArrayLike, sample_list: list[list[int]], degree:int=15, delta:int=6, max_scale:float=0.98, row_sparsity:int=None) -> QuantumCircuit:
    ''' sends |i> |k> |l> |0> to (-1)^{1_{C(i.k)<l}} |0> 
    Real part controlled by new ancilla bit'''

    if row_sparsity is None:
        row_sparsity = rowsparsity(matrix)

    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)

    qsvt_circuit = rcount_qsvt_quick(matrix, sample_list, degree, delta, max_scale, row_sparsity)
    qsvt_circuit.name = r'$P_{QSVT}(W_c)$'
    
    qsvt_adj = qsvt_circuit.inverse()
    qsvt_adj.name = r'$P_{QSVT}(W_c)^{\dagger}$'

    qsvt_control = qsvt_circuit.control(ctrl_state=0)
    qsvt_control.name = r'$cP_{QSVT}(W_c)$'

    qsvtadj_control = qsvt_adj.control(ctrl_state=1)
    qsvtadj_control.name = r'$cP_{QSVT}(W_c)^{\dagger}$'
    
    qr_row = QuantumRegister(num_qubits,name='q_i')
    qr_sparse = QuantumRegister(sparse_bits,name='q_k')
    qr_l = QuantumRegister(num_qubits,name='q_l')
    ar = AncillaRegister(1,name='a') #For encoding
    real_ancilla = AncillaRegister(1,name='r') #To get real part of QSVT function

    phase_oracle_circuit = QuantumCircuit(qr_row,qr_sparse,qr_l,ar,real_ancilla)

    phase_oracle_circuit.h(real_ancilla)

    phase_oracle_circuit.compose(qsvt_control,real_ancilla[:]+qr_row[:]+qr_sparse[:]+qr_l[:]+ar[:], inplace=True)
    phase_oracle_circuit.compose(qsvtadj_control,real_ancilla[:]+qr_row[:]+qr_sparse[:]+qr_l[:]+ar[:], inplace=True)

    phase_oracle_circuit.h(real_ancilla)
    
    return phase_oracle_circuit

def rcumulative_count_oracle_quick(matrix: ArrayLike, sample_list: list[list[float]], degree:int=17, delta:int=6, max_scale:float=0.95, row_sparsity:int=None) -> QuantumCircuit:
    '''Generates circuit O_c for specified matrix, given a list of samples'''

    if row_sparsity == None:
        row_sparsity = rowsparsity(matrix)
        
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)

    qr_row = QuantumRegister(num_qubits,name='q_i')
    qr_sparse = QuantumRegister(sparse_bits,name='q_k')
    qr_l = QuantumRegister(num_qubits,name='q_l')
    control_ancilla = AncillaRegister(1,name='o') #ancilla for controlled V_c, encodes 1[C(i,l)<k]
    ar = AncillaRegister(2,name='a') #Real part + encoding ancilla

    phase_oracle_circ = rcount_phase_oracle_quick(matrix=matrix, sample_list=sample_list, degree=degree, delta=delta, max_scale=max_scale, row_sparsity=row_sparsity)
    phase_oracle_circ.name = r'$V_c$'

    controlled_VC = phase_oracle_circ.control(1)
    controlled_VC.label = r'$cV_c$'

    cumulative_count_circuit = QuantumCircuit(qr_row,qr_sparse,qr_l,control_ancilla,ar)
    cumulative_count_circuit.x(control_ancilla)
    cumulative_count_circuit.h(control_ancilla)
    cumulative_count_circuit.compose(controlled_VC,control_ancilla[:]+qr_row[:]+qr_sparse[:]+qr_l[:]+ar[:],inplace=True)
    cumulative_count_circuit.h(control_ancilla)
    cumulative_count_circuit.x(control_ancilla)

    return cumulative_count_circuit

def row_index_oracle_quick(matrix: ArrayLike, sample_list: list[list[float]], degree:int=17, delta:int=6, max_scale:float=0.95, row_sparsity:int=None) -> QuantumCircuit:
    ''' sends |i> |k> |0> |0> to |i> |k> |j(i,k)> |0>
    '''
    if row_sparsity is None:
        row_sparsity = rowsparsity(matrix)
    sparsebits = required_sparse_bits(matrix)
    num_qubits = required_qubits(matrix)

    count_oracle_circuit = rcumulative_count_oracle_quick(matrix,sample_list=sample_list, 
    degree=degree, delta=delta, max_scale=max_scale, row_sparsity=row_sparsity)

    q_row = QuantumRegister(num_qubits,name='q_i')
    q_sparse = QuantumRegister(sparsebits,name='q_k')
    q_l = QuantumRegister(num_qubits,name='q_l')
    control_ancilla = AncillaRegister(1,name='o') #Ancilla for controlled V_c, encodes 1[C(i,l)<k]
    extra_ancillas = AncillaRegister(2,name='a') #Ancillas for U_c

    index_oracle_circ = QuantumCircuit(q_row,q_sparse,q_l,control_ancilla,extra_ancillas)

    for m in range(num_qubits):
        bit_index = num_qubits - m - 1 #start with most significant bit for binary search
        index_oracle_circ.x(q_l[bit_index])
        index_oracle_circ.compose(count_oracle_circuit,inplace=True)
        index_oracle_circ.x(q_l[bit_index])
        index_oracle_circ.swap(q_l[bit_index],control_ancilla[0])
        index_oracle_circ.barrier()

    # Clean k register
    # Iterate over bits of k, swap with l
    index_oracle_circ.x(control_ancilla[0])
    index_oracle_circ.compose(subtract_one_circuit(sparsebits),q_sparse[:], inplace=True) # k -> k-1 

    # The following leaves us with |i> |0> |j(i,k)> |0>_o
    for m in range(sparsebits):
        index_oracle_circ.swap(q_sparse[m],control_ancilla[0])
        index_oracle_circ.compose(count_oracle_circuit,inplace=True)
        index_oracle_circ.x(q_sparse[m])
    index_oracle_circ.x(control_ancilla[0]) 

    #swap qubits in l register past k register so that we have |i> |j(i,k)> |0> |0>_o
    for k in range(sparsebits):
        for l in range(num_qubits):
            index_oracle_circ.swap([num_qubits+(sparsebits-k-1)+l],[num_qubits+(sparsebits-k-1)+l+1])

    return index_oracle_circ

def col_index_oracle_quick(matrix: ArrayLike, sample_list: list[list[float]], degree:int=17, delta:int=6, max_scale:float=0.95, col_sparsity:int=None) -> QuantumCircuit:
    transpose_sample_list = sample_list.copy()
    for i in range(len(transpose_sample_list)):
        transpose_sample_list[i][0], transpose_sample_list[i][1] = transpose_sample_list[i][1], transpose_sample_list[i][0] 
    return row_index_oracle_quick(matrix.T, sample_list, degree, delta, max_scale, row_sparsity=col_sparsity)

#functions for matrix and sample generation
def generate_random_sparse(matrix_dimension:int, bitlength:int, sparsity:float, seed=None):
    #build our random sparse matrix
    np.random.seed(seed)

    sparse_matrix = np.zeros((matrix_dimension,matrix_dimension))

    row_sparsity = sparsity #proportion of nonzero entries in each row
    nonzero_entry_number = int(math.floor(row_sparsity*matrix_dimension))

    #add non-zero entries matrix
    row_nonzero_counts = np.count_nonzero(sparse_matrix, axis=1)
    col_nonzero_counts = np.count_nonzero(sparse_matrix , axis=0)

    for j in range(matrix_dimension):
        while row_nonzero_counts[j]  < nonzero_entry_number:
            sparse_matrix[j][np.random.choice(np.arange(matrix_dimension))] = np.random.randint(2**bitlength)
            row_nonzero_counts = np.count_nonzero(sparse_matrix, axis=1)
    col_nonzero_counts = np.count_nonzero(sparse_matrix , axis=0)
    for k in range(matrix_dimension):
        while col_nonzero_counts[k]  > nonzero_entry_number: #delete entries that make column too dense
            sparse_matrix[np.random.choice(np.arange(matrix_dimension))][k] = 0
            col_nonzero_counts = np.count_nonzero(sparse_matrix, axis=0)
    row_nonzero_counts = np.count_nonzero(sparse_matrix, axis=1)

    sparse_matrix = sparse_matrix.astype(int)
    return sparse_matrix

def sparse_element_stream(sparse_matrix,seednum=None) -> list[int,int,float]:
    ''' Uniformly randomly picks an index (i,j) such that A_{ij} != 0
    and returns a single sample [i, j, A_{ij}]'''
    np.random.seed(seednum)
    support_size = np.count_nonzero(sparse_matrix)
    support_indices = np.nonzero(sparse_matrix.astype(int))

    index = np.random.randint(0,support_size)
    sample=[int(support_indices[0][index]),int(support_indices[1][index]),sparse_matrix[support_indices[0][index],support_indices[1][index]]]
    return sample

def cumulative_count(sparse_matrix: ArrayLike) -> NDArray:
    '''Returns array containing all C(i,l) as defined by
    C(i,l) = |{j : A_{ij} != 0 , j < l}| for input matrix A'''
    cumulative_count = np.zeros_like(sparse_matrix)
    for i in range(len(sparse_matrix)):
        running_total = 0
        for l in range(1,len(sparse_matrix)): #C(i,0) == 0
            if sparse_matrix[i,l-1] != 0:
                running_total +=1
            cumulative_count[i,l] = running_total
    return cumulative_count

def get_samples(matrix: ArrayLike, sample_size:int=10, seednumber:int=None, samplerelative:bool=False) -> list[list[int]]:
    ''' Generates samples of a matrix for use in phase oracles'''
    if samplerelative == True:
        K = np.count_nonzero(matrix) #size of sample space
        sample_size = int(K*sample_size)
    
    if seednumber == None:
        seednumber = np.random.randint(1e8)

    sample_list = []

    seed = seednumber

    for _ in range(sample_size):
        sample_list.append(sparse_element_stream(matrix,seednum=seed))
        seed += 1

    return sample_list

# Helpful functions for building circuits from matrix
def rowsparsity(matrix: ArrayLike) -> int:
    return int(np.max(np.count_nonzero(matrix, axis=1)))

def required_qubits(matrix: ArrayLike) -> int:
    return int(math.ceil(np.log2(len(matrix))))

def required_sparse_bits(matrix: ArrayLike) -> int:
    return int(math.ceil(np.log2(rowsparsity(matrix)+1)))

def block_ancilla_index(matrix: ArrayLike, row_sparsity:int=None) -> int:
    ''' Given a matrix, computed number of qubits and sparisty
    to give index of block coding ancilla'''
    if row_sparsity == None:
        row_sparsity = rowsparsity(matrix)
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)
    return num_qubits*2 + sparse_bits

#functions for demos/ verification
def guess_matrix(sparsematrix,samples=25,startingseed=0,samplerelative=False, rounding=None):
    
    matrix_dim = len(sparsematrix)
    num_qubits = int(np.log2(len(sparsematrix)))

    reconstructed_matrix = np.zeros((matrix_dim,matrix_dim))
    reconstructed_variance = np.zeros((matrix_dim,matrix_dim))
    reconstructed_probs = np.zeros((matrix_dim,matrix_dim))
    bitlength = math.ceil(np.log2(np.max(sparsematrix)+1))
    element_oracle_circuit = element_oracle_gate(sparsematrix,samples,startingseed,samplerelative=samplerelative)
    
    for i in range(2**num_qubits):
        for j in range(2**num_qubits):
            if sparsematrix[i][j] != 0:
                # display(f'i= {i} j= {j}')

                i_state = Statevector.from_int(i,2**num_qubits)
                j_state = Statevector.from_int(j,2**num_qubits)
                blank_entry = Statevector.from_int(0,2**bitlength)

                starting_state = blank_entry.tensor(j_state.tensor(i_state))
                # display('starting state=',starting_state.draw('latex'))

                psi= starting_state.evolve(element_oracle_circuit)
                # display('psi=',psi.draw('latex'))

                probs = psi.probabilities(range(2*num_qubits,2*num_qubits+bitlength))
                max_arg = np.argmax(probs)
                element_guess = max_arg

                expected_entry = (np.arange(len(probs)) *probs).sum()
                if rounding != None:
                    expected_entry = np.round(expected_entry,decimals=rounding)

                variance_entry = (probs * np.arange(len(probs))**2).sum() - expected_entry**2
                if rounding != None:
                    variance_entry = np.round(variance_entry,decimals=rounding) - expected_entry**2

                reconstructed_matrix[i][j] = expected_entry
                reconstructed_variance[i][j] = variance_entry
                reconstructed_probs[i][j] = psi.probabilities()[element_guess]
    return reconstructed_matrix, reconstructed_variance, reconstructed_probs    

def find_desired_phase(matrix:ArrayLike, i:int, k:int, l:int) -> float:
    ''' Returns the desired phase theta(i,k,l) for the given matrix, i,k,l
    given by pi*(C(i,l) - k + 1/2) / 2(s_r+1/2)'''
    row_sparsity = rowsparsity(matrix)
    Cil = cumulative_count(matrix)[i][l]
    return np.pi*(Cil - k +0.5)/(2*(row_sparsity+0.5))

def rcumulative_count_unitary_test(matrix: ArrayLike, samplenum:int=25, samplerelative:bool=False, startingseed:int=None) -> tuple[list[float],list[float]]:
    ''' Tests the accuracy of the cumulative count unitary for a given matrix and sample list
    Uses both norm**2 error and angle error to determine accuracy'''
    cumulative_count_matrix = cumulative_count(matrix)

    errors = []
    angle_errors =[]

    row_sparsity = rowsparsity(matrix)
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)

    qr_row = QuantumRegister(num_qubits,name='q_i')
    qr_sparse = QuantumRegister(sparse_bits,name='q_k')
    qr_l = QuantumRegister(num_qubits,name='q_l')
    ar = AncillaRegister(num_qubits,name='a') #For the Comparator

    quick_cumulative_circ = QuantumCircuit(qr_row,qr_sparse,qr_l,ar)
    quick_cumulative_gate = rcumulative_count_unitary(matrix,samplenum=samplenum, samplerelative=samplerelative, startingseed=startingseed)
    quick_cumulative_circ.append(quick_cumulative_gate,qr_row[:]+qr_sparse[:]+qr_l[:]+ar[:])

    B = 2**(num_qubits*2 + sparse_bits) ##encoding block size
    target_phase_const = np.pi/(2*(row_sparsity+0.5))
    allowable_angle_error = target_phase_const*0.5
        
    for i in range(2**num_qubits):
        for k in range(1,row_sparsity+1):
            for l in range(2**num_qubits):
                desired_phase = find_desired_phase(matrix,i,k,l)
                initial_i = Statevector.from_int(i,2**num_qubits)
                initial_k = Statevector.from_int(k,2**sparse_bits)
                initial_l = Statevector.from_int(l,2**num_qubits)
                blank_ar = Statevector.from_int(0,2**num_qubits)
                starting_state = blank_ar.tensor((initial_l.tensor((initial_k.tensor(initial_i)))))
                psitarg = np.exp(desired_phase*1j)*starting_state
                psi = starting_state.evolve(quick_cumulative_circ)

                errors.append(np.sum(np.abs(psi.data - psitarg.data)**2))

                psiindex = int(np.where(psi.probabilities() > 0.5)[0][0]) #find state most likely to be in
                targindex = int(np.where(psitarg.probabilities() > 0.5)[0][0]) # find index of |i>|k>|l>|0> state
                
                psi_angle = np.angle(psi[psiindex]) # phi if < psi | i,k,l > = e^{i*phi}
                target_angle = np.angle(psitarg[targindex])

                angle_error = np.angle(np.exp(np.abs(psi_angle - target_angle)*1j)) #To handle when one angle close to -pi and the other close to pi

                if angle_error > allowable_angle_error:
                    display(f'i= {i} k= {k} l={l} failed')
                    display(f'angle error > {allowable_angle_error}')
                    display('target=',psitarg.draw('latex'))
                    display('our statevector=',psi.draw('latex'))
                    display(f'angle = {psi_angle} \ntarget angle = {target_angle}')
                    break
                angle_errors.append(angle_error)

                rtol = 0.5 # To catch complete duds
                if np.allclose(psi.data, psitarg.data,rtol=rtol) == False:
                    print(f'i= {i} k= {k} l={l} failed')
                    display('target=',psitarg.draw('latex'))
                    display('our statevector=',psi.draw('latex'))
                    display(f'2-norm difference = {np.sum(np.abs(psi.data - psitarg.data)**2)}')

    print(f'''All tests for cumulative count Unitary passed for test matrix \n {matrix} \n
    average 2-norm error: {np.mean(errors)} \n max 2-norm error: {np.max(errors)} \n 
    average angle error: {np.mean(angle_errors)} \n max angle error: {np.max(angle_errors)}
    ''')
    return errors, angle_errors

def quick_unitary_test(matrix: ArrayLike, sample_list: list[list[float]]) -> tuple[list[float],list[float]]:
    ''' Tests the accuracy of the cumulative count unitary for a given matrix and sample list
    Uses both norm**2 error and angle error to determine accuracy'''
    cumulative_count_matrix = cumulative_count(matrix)

    errors = []
    angle_errors =[]

    row_sparsity = rowsparsity(matrix)
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)

    qr_row = QuantumRegister(num_qubits,name='q_i')
    qr_sparse = QuantumRegister(sparse_bits,name='q_k')
    qr_l = QuantumRegister(num_qubits,name='q_l')

    quick_cumulative_circ = QuantumCircuit(qr_row,qr_sparse,qr_l)
    quick_cumulative_gate = quick_rcumulative_count_unitary(matrix,sample_list)
    quick_cumulative_circ.append(quick_cumulative_gate,qr_row[:]+qr_sparse[:]+qr_l[:])

    B = 2**(num_qubits*2 + sparse_bits) ##encoding block size
    target_phase_const = np.pi/(2*(row_sparsity+0.5))
    allowable_angle_error = target_phase_const*0.5
        
    for i in range(2**num_qubits):
        for k in range(1,row_sparsity+1):
            for l in range(2**num_qubits):
                desired_phase = find_desired_phase(matrix,i,k,l)
                initial_i = Statevector.from_int(i,2**num_qubits)
                initial_k = Statevector.from_int(k,2**sparse_bits)
                initial_l = Statevector.from_int(l,2**num_qubits)
                starting_state = initial_l.tensor((initial_k.tensor(initial_i)))
                psitarg = np.exp(desired_phase*1j)*starting_state
                psi = starting_state.evolve(quick_cumulative_circ)

                errors.append(np.sum(np.abs(psi.data - psitarg.data)**2))

                psiindex = int(np.where(psi.probabilities() > 0.5)[0][0]) #find state most likely to be in
                targindex = int(np.where(psitarg.probabilities() > 0.5)[0][0]) # find index of |i>|k>|l>|0> state
                
                psi_angle = np.angle(psi[psiindex]) # phi if < psi | i,k,l > = e^{i*phi}
                target_angle = np.angle(psitarg[targindex])

                angle_error = np.angle(np.exp(np.abs(psi_angle - target_angle)*1j)) #To handle when one angle close to -pi and the other close to pi

                if angle_error > allowable_angle_error:
                    display(f'i= {i} k= {k} l={l} failed')
                    display(f'angle error > {allowable_angle_error}')
                    display('target=',psitarg.draw('latex'))
                    display('our statevector=',psi.draw('latex'))
                    display(f'angle = {psi_angle} \n target angle = {target_angle}')
                    break
                angle_errors.append(angle_error)

                rtol = 0.5 # To catch complete duds
                if np.allclose(psi.data, psitarg.data,rtol=rtol) == False:
                    print(f'i= {i} k= {k} l={l} failed')
                    display('target=',psitarg.draw('latex'))
                    display('our statevector=',psi.draw('latex'))
                    display(f'2-norm difference = {np.sum(np.abs(psi.data - psitarg.data)**2)}')

    print(f'''All tests for cumulative count Unitary passed for test matrix \n {matrix} \n
    average 2-norm error: {np.mean(errors)} \n max 2-norm error: {np.max(errors)} \n 
    average angle error: {np.mean(angle_errors)} \n max angle error: {np.max(angle_errors)}
    ''')
    return errors, angle_errors

def quick_encoding_check(matrix: ArrayLike, sample_list: list[list[float]]) -> list[float]:
    ''' Tests the accuracy of the cumulative count block encoder for a given matrix and sample list
    Measure 2-norm error of state vectors on the encoding block'''
    errors =[]
    row_sparsity = rowsparsity(matrix)
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)
    target_phase_const = np.pi/(2*(row_sparsity+0.5))

    B = 2**(num_qubits*2 + sparse_bits) #encoding block size for |0>_a

    encoding_circuit = rcumulative_encoding_quick(matrix=matrix, sample_list=sample_list)
    for i in range(2**num_qubits):
        for k in range(1,row_sparsity+1):
            for l in range(2**num_qubits):
                desired_factor = np.sin(find_desired_phase(matrix,i,k,l))

                initial_i = Statevector.from_int(i,2**num_qubits)
                initial_k = Statevector.from_int(k,2**sparse_bits)
                initial_l = Statevector.from_int(l,2**num_qubits)
                blank_ar = Statevector.from_int(0,2**1) #encoding ancilla
                starting_state = blank_ar.tensor((initial_l.tensor((initial_k.tensor(initial_i)))))
                psi = starting_state.evolve(encoding_circuit)
                psitarg = desired_factor*starting_state

                errors.append(np.sum(np.abs(psi.data[0:B] - psitarg.data[0:B])**2))

                rtol = 0.5 # checking for when result is way off 
                if np.allclose(psi.data[0:B], psitarg.data[0:B], rtol=rtol) == False:
                    display(f'i= {i} k= {k} l={l} failed')                
                    display('target=',psitarg.draw('latex'))
                    display('our statevector=',psi.draw('latex'))
                    display('initiated as ',starting_state.draw('latex'))
                    display(f'absolute difference = {np.sum(np.abs(psi.data[0:B] - psitarg.data[0:B]))}')

    print(f'''All tests for cumulative count block encoder passed for test matrix \n {matrix} \n
    average 2-norm error: {np.mean(errors)} \n max 2-norm error: {np.max(errors)}
    ''')
    return errors

def quick_rcount_phase_check(matrix: ArrayLike, sample_list: list[list[float]], degree:int=15, delta:int=6, max_scale:float=0.98, row_sparsity:int=None) -> tuple[list[float],list[float]]:
    '''Checks accuracy of the Phase Oracle PSVT(W_c) for a given matrix and sample list.
    Computes 2-norm error and angle error across all basis states |i>|k>|l>|0>'''
    errors =[]
    angle_errors = []

    rcount_oracle_circuit = rcount_phase_oracle_quick(matrix, sample_list, degree, delta, max_scale, row_sparsity)

    row_sparsity = rowsparsity(matrix)
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)
    target_phase_const = np.pi/(2*(row_sparsity+0.5))
    B = 2**(num_qubits*2 + sparse_bits) #encoding block size for |0>_a

    for i in range(2**num_qubits):
        for k in range(1,row_sparsity+1):
            for l in range(2**num_qubits):
                desired_factor = np.sign(find_desired_phase(matrix,i,k,l))

                initial_i = Statevector.from_int(i,2**num_qubits)
                initial_k = Statevector.from_int(k,2**sparse_bits)
                initial_l = Statevector.from_int(l,2**num_qubits)
                blank_ar = Statevector.from_int(0,2**1) #encoding ancilla
                blank_real = Statevector.from_int(0,2**1) #control for getting real part
                starting_state = blank_real.tensor(blank_ar.tensor((initial_l.tensor((initial_k.tensor(initial_i))))))
                psi = starting_state.evolve(rcount_oracle_circuit)
                psitarg = desired_factor*starting_state

                #A lot of mass may end up on block(s) corresponding to |1>_a, but we just care about 1 - P(measure 1_{C(i,j)<k}) < epsilon when in a |0>_a state.
                normpsi = psi.data[0:B]/np.linalg.norm(psi.data[0:B])
                psiblock = Statevector(normpsi)
                normpsitarg = psitarg.data[0:B]/np.linalg.norm(psitarg.data[0:B])
                psitargblock = Statevector(normpsitarg)

                psiindex = int(np.where( psiblock.probabilities() > 0.5)[0][0])
                targindex = int(np.where( psitargblock.probabilities() > 0.5)[0][0]) # find index of |i>|k>|l>|0> state
                
                abs_error = np.sum(np.abs((normpsi) - normpsitarg)**2) # 2-norm error between the two vectors
                errors.append(abs_error)

                psi_angle = np.angle(psi[psiindex]) # phi if < psi | i,k,l > = e^{i*phi}
                target_angle = np.angle(psitarg[targindex])
                angle_error = np.abs(np.angle(np.exp(np.abs(psi_angle - target_angle)*1j))) #To handle when one angle close to -pi and the other close to pi
                angle_errors.append(angle_error)

    print(f'''All tests concluded for Cumulative Count Phase Oracle for test matrix \n {matrix} \n
    average error in real part: {np.mean(errors)} \n max error: {np.max(errors)} \n
    average error in angle: {np.mean(angle_errors)} \n max error in angle: {np.max(angle_errors)} \n
    ''')
    return errors, angle_errors  

def quick_rcumulative_oracle_check(matrix: ArrayLike, sample_list: list[list[float]], degree:int=17, delta:int=6, max_scale:float=0.95, row_sparsity:int=None) -> list[float]:
    '''Checks accuracy of the Cumulative Count Oracle O_c for a given matrix and sample list.
    Computes 2-norm error across all basis states |i>|k>|l>|0>.
    Errors are in form P(ancilla bit = NOT(C(i,l)<k))'''

    errors =[]
    rcount_oracle_circuit = rcumulative_count_oracle_quick(matrix, sample_list, degree, delta, max_scale, row_sparsity)

    row_sparsity = rowsparsity(matrix)
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)
    target_phase_const = np.pi/(2*(row_sparsity+0.5))
    B = 2**(num_qubits*2 + sparse_bits +1) #encoding block size for |0>_a: our new XOR ancilla is before encoding ancilla
    count_matrix = cumulative_count(matrix)

    for i in range(2**num_qubits):
        for k in range(1,row_sparsity+1):
            for l in range(2**num_qubits):
                desired_bit = int(count_matrix[i][l] < k)

                initial_i = Statevector.from_int(i,2**num_qubits)
                initial_k = Statevector.from_int(k,2**sparse_bits)
                initial_l = Statevector.from_int(l,2**num_qubits)
                blank_xor = Statevector.from_int(0,2**1) #encoding ancilla
                answer_xor = Statevector.from_int(desired_bit,2**1) #encoding ancilla
                blank_ar = Statevector.from_int(0,2**2) #encoding ancilla
                starting_state = blank_ar.tensor(blank_xor.tensor((initial_l.tensor((initial_k.tensor(initial_i))))))
                psi = starting_state.evolve(rcount_oracle_circuit)
                psitarg = blank_ar.tensor(answer_xor.tensor((initial_l.tensor((initial_k.tensor(initial_i))))))

                # Don't need indices with ancillas
                normpsi = psi.data[0:B]/np.linalg.norm(psi.data[0:B])
                psiblock = Statevector(normpsi)
                normpsitarg = psitarg.data[0:B]/np.linalg.norm(psitarg.data[0:B])
                psitargblock = Statevector(normpsitarg)
                targindex = int(np.where(psitarg.probabilities() >0.9)[0][0]) # tells us which state we want to be in

                error = 1 - psiblock.probabilities()[targindex] # P( ancilla bit o = NOT(C(i,j)<k) )
                errors.append(error)
                if error > 0.5:
                    display('i= ', i, ' k= ', k, ' l=', l)
                    display('C(i,l)= ', count_matrix[i][l])
                    display('desired statevector=',psitargblock.draw('latex'))
                    display('our statevector=',psiblock.draw('latex'))
                    display('error=',error)

    print(f'''All tests concluded for Row Cumulative Count Oracle for test matrix \n {matrix} \n
    average prob that measurement of ancilla bit is in the wrong state: {np.mean(errors)} \n max prob: {np.max(errors)} \n
    ''')
    return errors

def rowindexquick_test(matrix,sample_list,degree,delta,max_scale,row_sparsity=None,prints=True, buildmatrix=True):
    ''' Tests the accuracy of the cumulative count unitary for a given matrix and sample list
    Makes best guess for j(i,k) and records corresponding probability'''
    guess_list = []
    probs_list = []
    matrix_guess = []
    nonzeros = np.nonzero(matrix)
    if buildmatrix == True:
            matrix_guess = np.zeros_like(matrix)

    if row_sparsity is None:
        row_sparsity = rowsparsity(matrix)
    num_qubits = required_qubits(matrix)
    sparse_bits = required_sparse_bits(matrix)

    q_row = QuantumRegister(num_qubits,name='q_i')
    q_sparse = QuantumRegister(sparse_bits,name='q_k')
    q_l = QuantumRegister(num_qubits,name='q_l')
    control_ancilla = AncillaRegister(1,name='o') #Ancilla for controlled V_c, encodes 1[C(i,l)<k]
    extra_ancillas = AncillaRegister(2,name='a') #Ancillas for U_c

    row_index_oracle_circuit = row_index_oracle_quick(matrix,sample_list,degree,delta,max_scale,row_sparsity)

    initial_l = Statevector.from_int(0,2**num_qubits) # l = 0
    blank_xor = Statevector.from_int(0,2**1) #encoding ancilla
    blank_ar = Statevector.from_int(0,2**2) #encoding ancilla

    blank_state = blank_ar.tensor(blank_xor.tensor(initial_l))
    for i in range(2**num_qubits):
        nonzero_row = np.where(nonzeros[0] == i)
        for k in range(1,row_sparsity+1):
            initial_i = Statevector.from_int(i,2**num_qubits)
            initial_k = Statevector.from_int(k,2**sparse_bits)

            starting_state = blank_state.tensor((initial_k.tensor(initial_i)))
            psi = starting_state.evolve(row_index_oracle_circuit)

            B = 2**(num_qubits*2 + sparse_bits + 1)

            normpsi = psi.data[0:B]/np.linalg.norm(psi.data[0:B])
            psiblock = Statevector(normpsi)

            guess_index = np.where(np.abs(psiblock.probabilities() - np.max(psiblock.probabilities())) < 0.01)

            helper_index = nonzero_row[0][k-1]
            correct_index = nonzeros[1][helper_index] #picks out j(i,k)
            # display(f'helper index is {helper_index} and correct index is {correct_index}')

            successprob = psiblock.probabilities()[i + 2**num_qubits*correct_index] #prob of guess being correct
            guess_j = guess_index[0][0] // (2**(num_qubits))
            # display(f'success prob index is {2**num_qubits*correct_index}')
            # display(f'success probability is {successprob} and guess is {guess_j}')

            guess_list.append(int(guess_j))
            probs_list.append(successprob)
            if buildmatrix == True:
                matrix_guess[i][guess_j] = 1

    correct_matrix = np.where(matrix > 1e-10, 1, 0)

    if prints == True:
        print(f'''Using row index oracle, guessing support of \n {matrix} \n to be... \n {matrix_guess}''')

        if np.allclose(matrix_guess,correct_matrix) == True:
            print('Correctly guessed matrix!')
    
    return guess_list, probs_list, matrix_guess

def colindexquick_test(matrix: ArrayLike, sample_list: list[list[float]], degree:int=17, delta:int=6, max_scale:float=0.95, col_sparsity:int=None) -> tuple[list[float],list[float],NDArray]:
    ''' Tests the accuracy of the cumulative count unitary for a given matrix and sample list
    Makes best guess for j(i,k) and records corresponding probability'''
    guess_list = []
    probs_list = []

    matrix_guess = np.zeros_like(matrix)

    if col_sparsity is None:
        col_sparsity = rowsparsity(matrix.T)
    num_qubits = required_qubits(matrix.T)
    sparse_bits = required_sparse_bits(matrix.T)

    q_row = QuantumRegister(num_qubits,name='q_j')
    q_sparse = QuantumRegister(sparse_bits,name='q_k')
    q_l = QuantumRegister(num_qubits,name='q_l')
    control_ancilla = AncillaRegister(1,name='o') #Ancilla for controlled V_c, encodes 1[C(i,l)<k]
    extra_ancillas = AncillaRegister(2,name='a') #Ancillas for U_c

    row_index_oracle_circuit = col_index_oracle_quick(matrix,sample_list,degree,delta,max_scale,col_sparsity)

    initial_l = Statevector.from_int(0,2**num_qubits) # l = 0
    blank_xor = Statevector.from_int(0,2**1) #encoding ancilla
    blank_ar = Statevector.from_int(0,2**2) #encoding ancilla
    blank_state = blank_ar.tensor(blank_xor.tensor(initial_l))

    for j in range(2**num_qubits):
        for k in range(1,col_sparsity+1):
            initial_i = Statevector.from_int(j,2**num_qubits)
            initial_k = Statevector.from_int(k,2**sparse_bits)

            starting_state = blank_state.tensor((initial_k.tensor(initial_i))) #|i>|k>|0>|0>
            psi = starting_state.evolve(row_index_oracle_circuit)

            B = 2**(num_qubits*2 + sparse_bits + 1)

            normpsi = psi.data[0:B]/np.linalg.norm(psi.data[0:B])
            psiblock = Statevector(normpsi)

            guess_index = np.where(np.abs(psiblock.probabilities() - np.max(psiblock.probabilities())) < 0.01)
 
            confidence = psiblock.probabilities()[guess_index]
            guess_i = guess_index[0][0] // (2**(num_qubits))
            # print(f'j({i},{k}) = {guess_j} \n with probability: {confidence[0]}')

            guess_list.append(int(guess_i))
            probs_list.append(confidence[0])
            matrix_guess[guess_i][j] = 1

    print(f'''Using column index oracle, guessing support of \n {matrix} \n to be... \n {matrix_guess}''')

    correct_matrix = np.where(matrix > 1e-10, 1, 0)
    if np.allclose(matrix_guess,correct_matrix) == True:
        print('Correctly guessed matrix!')
    
    return guess_list, probs_list, matrix_guess

#arithmetic functions
def subtract_one_circuit(bitlength:int) -> QuantumCircuit:
    ''' Returns gate that subtracts 1 from all basis states
    sends 0 all 1s'''
    subtractor_circuit = QuantumCircuit(bitlength)
    subtractor_circuit.x([0])
    for i in range(1,bitlength):
        subtractor_circuit.mcx(list(range(i)),[i])
    subtractor_circuit.name = r'Subtract $1$'
    return subtractor_circuit
    
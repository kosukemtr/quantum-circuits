from qulacs import Observable
from qulacs import QuantumState, QuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge
from numpy import polyfit
import numpy as np
import matplotlib.pyplot as plt


def convert_openfermion_op(n_qubit, openfermion_op):
    """convert_openfermion_op
    Args:
        n_qubit (:class:`int`)
        openfermion_op (:class:`openfermion.ops.QubitOperator`)
    Returns:
        :class:`qulacs.Observable`
    """
    ret = Observable(n_qubit)

    for pauli_product in openfermion_op.terms:
        coef = float(np.real(openfermion_op.terms[pauli_product]))
        pauli_string = ''
        for pauli_operator in pauli_product:
            pauli_string += pauli_operator[1] + ' ' + str(pauli_operator[0])
            pauli_string += ' '
        ret.add_operator(coef, pauli_string[:-1])

    return ret


def ifcommutes(pauli_string1, pauli_string2):
    """ifcommutes
    Args:
        pauli_string1 (str):
            pauli string such as "X 0 Y 1 Y 2 X 3"
        psuli_string2 (str)
    Return:
        :float: sampled expectation value of the observable
    """
    pauli_id1 = pauli_string1.split(" ")[::2]
    pauli_index1 = np.array(list(map(int, pauli_string1.split(" ")[1::2])))
    pauli_id2 = pauli_string2.split(" ")[::2]
    pauli_index2 = np.array(list(map(int, pauli_string2.split(" ")[1::2])))
    flag = 1
    for k, index in enumerate(pauli_index1):
        if index in pauli_index2:
            # if single pauli operator does not commute
            if pauli_id1[k] != pauli_id2[int(np.where(pauli_index2 == index)[0])]:
                flag *= -1

    if flag == -1:
        return False
    else:
        return True


def sample_observable(state, obs, n_sample):
    """
    Args:
        state (qulacs.QuantumState):
        obs (qulacs.Observable)
        n_sample (int):  number of samples for each observable
    Return:
        :float: sampled expectation value of the observable
    """
    n_term = obs.get_term_count()
    n_qubit = obs.get_qubit_count()

    pauli_terms = [obs.get_term(i) for i in range(n_term)]
    coefs = [p.get_coef() for p in pauli_terms]
    pauli_ids = [p.get_pauli_id_list() for p in pauli_terms]
    pauli_indices = [p.get_index_list() for p in pauli_terms]

    exp = 0
    measured_state = state.copy()
    for i in range(n_term):
        state.load(measured_state)
        measurement_circuit = QuantumCircuit(n_qubit)
        if len(pauli_ids[i]) == 0:  # means identity
            exp += coefs[i]
            continue
        mask = ''.join(['1' if n_qubit-1-k in pauli_indices[i]
                        else '0' for k in range(n_qubit)])
        for single_pauli, index in zip(pauli_ids[i], pauli_indices[i]):
            if single_pauli == 1:
                measurement_circuit.add_H_gate(index)
            elif single_pauli == 2:
                measurement_circuit.add_Sdag_gate(index)
                measurement_circuit.add_H_gate(index)
        measurement_circuit.update_quantum_state(state)
        samples = state.sampling(n_sample)
        mask = int(mask, 2)
        exp += coefs[i]*sum(list(map(lambda x: (-1) **
                                     (bin(x & mask).count('1')), samples)))/n_sample

    return exp


def sample_observable_noisy_circuit(circuit, initial_state, obs,
                                    n_circuit_sample=1000,
                                    n_sample_per_circuit=1):
    """
    Args:
        circuit (:class:`qulacs.QuantumCircuit`)
        initial_state (:class:`qulacs.QuantumState`)
        obs (:class:`qulacs.Observable`)
        n_circuit_sample (:class:`int`):  number of circuit samples
        n_sample (:class:`int`):  number of samples per one circuit samples
    Return:
        :float: sampled expectation value of the observable
    """
    n_term = obs.get_term_count()
    n_qubit = obs.get_qubit_count()

    pauli_terms = [obs.get_term(i) for i in range(n_term)]
    coefs = [p.get_coef() for p in pauli_terms]
    pauli_ids = [p.get_pauli_id_list() for p in pauli_terms]
    pauli_indices = [p.get_index_list() for p in pauli_terms]

    exp = 0
    state = initial_state.copy()
    for c in range(n_circuit_sample):
        state.load(initial_state)
        circuit.update_quantum_state(state)
        for i in range(n_term):
            measurement_circuit = QuantumCircuit(n_qubit)
            if len(pauli_ids[i]) == 0:  # means identity
                exp += coefs[i]
                continue
            mask = ''.join(['1' if n_qubit-1-k in pauli_indices[i]
                            else '0' for k in range(n_qubit)])
            for single_pauli, index in zip(pauli_ids[i], pauli_indices[i]):
                if single_pauli == 1:
                    measurement_circuit.add_H_gate(index)
                elif single_pauli == 2:
                    measurement_circuit.add_Sdag_gate(index)
                    uncompute_circuit.add_H_gate(index)
            measurement_circuit.update_quantum_state(state)
            samples = state.sampling(n_sample_per_circuit)
            mask = int(mask, 2)
            exp += coefs[i]*sum(list(map(lambda x: (-1)**(bin(x &
                                                              mask).count('1')), samples)))/n_sample_per_circuit
    exp /= n_circuit_sample
    return exp


def error_mitigation_sample(circuit_list, error_list, initial_state, obs,
                            n_circuit_sample=1000,
                            n_sample_per_circuit=1,
                            mode="linear",
                            return_full=False):
    """error_mitigation_sample

    Args:
        circuit_list (:class:`list` of `qulacs.QuantumCircuit` or `qulacs.ParametricQuantumCircuit`):
            list of quantum circuit with deifferent error rate
        initial_state (:class:`qulacs.QuantumState`):
            list of quantum circuit with deifferent error rat
        error_list (:class:`list`):
            list of error rate for each quantum circuit
        obs (:class:`qulacs.Observable`):
            measured observable
        n_circuit_sample (:class:`int`):
            number of circuit samples
        n_sample_per_circuit (:class:`int`):
            number of sample per circuit
        mode (:class:`str`):
            "linear" = linear extraplation
            "exp" = exponential extrapolation

    Returns:
        :class:`float` when return full is False (default)
        :class:`tuple` of (mitigated output, list of sampled expectation values, fit coefficients)
         when return full is True
    """
    exp_array = []
    for circuit in circuit_list:
        exp = sample_observable_noisy_circuit(circuit, initial_state, obs,
                                              n_circuit_sample=n_circuit_sample,
                                              n_sample_per_circuit=n_sample_per_circuit)
        exp_array.append(exp)
    if mode == "linear":
        fit_coefs = polyfit(error_list, exp_array, 1)
    elif mode == "exp":
        fit_coefs, _ = curve_fit(
            lambda x, a, b: a*np.exp(b*x) + bias, error_list, exp_array)

    if return_full:
        return fit_coefs[1], exp_array, fit_coefs
    else:
        return fit_coefs[1]

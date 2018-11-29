import vqe_utils
from scipy.optimize import minimize
from vqe_utils import sample_observable_noisy_circuit
from vqe_utils import error_mitigation_sample
from numpy import pi, zeros


class VqeOptimizer(object):
    """docstring for VqeOptimizer
    Args:
        parametric_circuit_list (:class:`list` of `qulacs.ParametricCircuit`):
            list of quantum circuit with different error rate.
        initial_state (:class:`qulacs.QuantumState`):
            initial state of the quatum circuit
        observable (:class:`qulacs.Observable`):
            the Hamiltonian to be minimized
        initial_param (:class:`numpy.ndarray`):
            initial parameter for the optimization
        error_list (:class:`numpy.ndarray`):
            error rate corresponding to each quantum circuit
            (Default is None for the case where we use the quantum circuit without noise)
        error_mitigation_mode (:class:``str):
            "linear": linear extrapolation
            "exp": exponetial extrapolation
            "exact": exact expectation value (use it only for non-noisy case)
            "none": does not perform error mitigation 
                    (quantum circuit with the lowest noise will be used in the optimization)
        n_circuit_sample (:class:`int`):
            specifies how many noisy circuit are used.
        n_sample_per_circuit (:class:`int`):
            specifies how many samples (per each circuit per each terms of observable)
            are taken to evaluate expectation value.
        n_circuit_sample_grad (:class:`int`):
            circuit samples for gradient evaluation
        n_sample_per_circuit_grad (:class:`int`):
            sample per circuit for gradient evaluation
        optimization_mode (:class:`str`):
            "BFGS": BFGS method from scipy
            "Nelder-Mead": Nelder Mead method from scipy
            "adam": Adam 
        ifexact (:class:`bool`):
            if True, the optimization is performed with exact expectation value
        options (:class:`dict`):
            options that are passed to optimization algorithm
        noiseless_circuit (:class:`qulacs.ParametricQuantumCircuit`):
            circuit without noise for the exact evaluaion of expectation value

    Attributes:
        parametric_circuit_list (:class:`list` of `qulacs.ParametricCircuit`)
        initial_state (:class:`qulacs.QuantumState`)
        observable (:class:`qulacs.Observable`)
        initial_param (:class:`numpy.ndarray`)
        error_list (:class:`numpy.ndarray`)
        error_mitigation_mode (:class:`str`)
        n_circuit_sample (:class:`int`)
        n_sample_per_circuit (:class:`int`)
        optimization_mode (:class:`str`)
        exp_history (:class:`list` of float):
            stores history during optimization
        parameter_histroy (:class:`list` of np.ndarray):
            stores history during optimization
        n_sample_per_iteration (:class:`list` of int):
            stores the number of samples taken in each iteration
        _n_predsent_sample (:class:`int`):
            (private) present number of samples
        _present_target_func (:class:`float`):
            (private) stores the value of target function at the last evaluation
    """

    def __init__(self, parametric_circuit_list, initial_state,
                 observable, initial_param,
                 error_list, error_mitigation_mode="linear",
                 n_circuit_sample=1000, n_sample_per_circuit=1,
                 optimization_mode="BFGS",
                 ifexact=False,
                 noiseless_circuit=None,
                 options=None):
        self.parametric_circuit_list = parametric_circuit_list
        self.initial_state = initial_state
        self.observable = observable
        self.initial_param = initial_param
        self.error_list = error_list
        self.error_mitigation_mode = error_mitigation_mode
        self.n_circuit_sample = n_circuit_sample
        self.n_circuit_sample_grad = n_circuit_sample
        self.n_sample_per_circuit = n_sample_per_circuit
        self.n_sample_per_circuit_grad = n_sample_per_circuit
        self.optimization_mode = optimization_mode
        self.ifexact = ifexact
        self.noiseless_circuit = noiseless_circuit
        self.exp_history = []
        self.parameter_history = []
        self.n_sample_per_iteration_history = []
        self._n_present_sample = 0
        self._present_target_func = 0

    def _callback(self, param):
        """callback
        callback function for each iteration
        """
        self.parameter_history.append(param)
        self.exp_history.append(self._present_target_func)
        self.n_sample_per_iteration_history.append(self._n_present_sample)
        self._n_present_sample = 0

    def optimize(self):
        """optimize
        run VQE.
        """
        if not self.ifexact:
            target_func = self._target_func_sample
            grad = self._gradient_sample
        else:
            target_func = self._target_func_exact
            grad = self._gradient_exact
        self.exp_history.append(target_func(self.initial_param))
        self.parameter_history.append(self.initial_param)
        if self.optimization_mode == "BFGS":
            opt = minimize(target_func, self.initial_param,
                           method="BFGS", jac=grad, callback=self._callback)
        if self.optimization_mode == "CG":
            opt = minimize(target_func, self.initial_param,
                           method="CG", jac=grad, callback=self._callback)
        return opt.x

    def _target_func_sample(self, param):
        """_target_func
        target function for the optimization

        Args:
            param (:class:`numy.ndarray`):
                circuit parameter
        Return:
            :class:`float`
        """
        ret = self.sample_mitigated_output(param,
                                           self.n_circuit_sample,
                                           self.n_sample_per_circuit,
                                           mode=self.error_mitigation_mode,
                                           )
        self._present_target_func = ret
        self._n_present_sample += self.n_circuit_sample * \
            self.n_sample_per_circuit * \
            len(self.parametric_circuit_list)
        return ret

    def _gradient_sample(self, param):
        ret = zeros(len(param))
        for i in range(len(param)):
            param[i] += pi/4
            plus = self.sample_mitigated_output(param,
                                                self.n_circuit_sample_grad,
                                                self.n_sample_per_circuit_grad,
                                                mode=self.error_mitigation_mode
                                                )
            self._n_present_sample += self.n_circuit_sample_grad * \
                self.n_sample_per_circuit_grad * \
                len(self.parametric_circuit_list)
            param[i] -= pi/2
            minus = self.sample_mitigated_output(param,
                                                 self.n_circuit_sample_grad,
                                                 self.n_sample_per_circuit_grad,
                                                 mode=self.error_mitigation_mode
                                                 )
            self._n_present_sample += self.n_circuit_sample_grad * \
                self.n_sample_per_circuit_grad * \
                len(self.parametric_circuit_list)
            ret[i] += (plus - minus)/2
            param[i] += pi/4
        return ret

    def _target_func_exact(self, param):
        """_target_func
        target function for the optimization for exact evaluation of expectation value

        Args:
            param (:class:`numy.ndarray`):
                circuit parameter
        Return:
            :class:`float`
        """
        ret = self.exact_output(self, param)
        self._present_target_func = ret
        return ret

    def _gradient_exact(self, param):
        """_gradient_exact
        gradient for exact expectation value

        Args:
            param (:class:`numy.ndarray`):
                circuit parameter
        Return:
            :class:`float`
        """
        ret = zeros(len(param))
        for i in range(len(param)):
            param[i] += pi/4
            plus = self.exact_output(self, param)
            param[i] -= pi/2
            minus = self.exact_output(self, param)
            ret[i] += (plus - minus)/2
            param[i] += pi/4
        return ret

    def sample_mitigated_output(self, param,
                                n_circuit_sample=None,
                                n_sample_per_circuit=None,
                                mode=None,
                                return_full=False):
        """sample_mitigated_output

        Args:
            param (:class:`numpy.ndarray`):
                circuit parameter
            n_circuit_sample (:class:`int`):
                number of circuit samples
            n_sample_per_circuit (:class:`int`):
                number of sample per circuit
            mode (:class:`str`):
                "linear" = linear extraplation
                "exp" = exponential extrapolation
            return_full (:class:`bool`)
        Return:
            :class:`float` when return full is False (default)
            :class:`tuple` of (mitigated output, list of sampled expectation values, fit coefficients)
            when return full is True
        """
        if n_circuit_sample is None:
            n_circuit_sample = self.n_circuit_sample
        if n_sample_per_circuit is None:
            n_sample_per_circuit = self.n_sample_per_circuit
        if mode is None:
            mode = self.error_mitigation_mode

        # set parameter for all parametric circuit
        for circuit in self.parametric_circuit_list:
            for p_index, p in enumerate(param):
                circuit.set_parameter(p_index, p)

        output = error_mitigation_sample(self.parametric_circuit_list,
                                         self.error_list,
                                         self.initial_state,
                                         self.observable,
                                         n_circuit_sample=n_circuit_sample,
                                         n_sample_per_circuit=n_sample_per_circuit,
                                         mode=mode,
                                         return_full=return_full)
        return output

    def sample_output(self, param,
                      n_circuit_sample=None,
                      n_sample_per_circuit=None,
                      mode=None,
                      return_full=False):
        """output

        Args:
            param (:class:`numpy.ndarray`):
                circuit parameter
            n_circuit_sample (:class:`int`):
                number of circuit samples
            n_sample_per_circuit (:class:`int`):
                number of sample per circuit
            return_full (:class:`bool`)
        Return:
            :class:`float` when return full is False (default)
            :class:`tuple` of (mitigated output, list of sampled expectation values, fit coefficients)
            when return full is True
        """
        if n_circuit_sample is None:
            n_circuit_sample = self.n_circuit_sample
        if n_sample_per_circuit is None:
            n_sample_per_circuit = self.n_sample_per_circuit

        # set parameter for all parametric circuit
        for p_index, p in enumerate(param):
            self.parametric_circuit_list[0].set_parameter(p_index, p)

        output = sample_observable_noisy_circuit(self.parametric_circuit_list[0],
                                                 self.initial_state,
                                                 self.observable,
                                                 n_circuit_sample=n_circuit_sample,
                                                 n_sample_per_circuit=n_sample_per_circuit
                                                 )
        return output

    def exact_output(self, param):
        """exact_output

        Args:
            param (:class:`numpy.ndarray`):
                circuit parameter
        Return:
            :class:`float`
        """
        for p_index, p in enumerate(param):
            self.noiseless_circuit.set_parameter(p_index, p)

        state = self.initial_state.copy()
        self.noiseless_circuit.update_quantum_state(state)
        return self.observable.get_expectation_value(state)


def test_VqeOptimizer():
    from qulacs import ParametricQuantumCircuit
    from qulacs import QuantumState
    from qulacs import Observable
    from qulacs.gate import Probabilistic, X, Y, Z
    import numpy as np
    import matplotlib.pyplot as plt

    n_qubit = 2
    p_list = [0.05, 0.1, 0.15]
    parametric_circuit_list = \
        [ParametricQuantumCircuit(n_qubit)
         for i in range(len(p_list))]
    initial_state = QuantumState(n_qubit)

    for (p, circuit) in zip(p_list, parametric_circuit_list):
        circuit.add_H_gate(0)
        circuit.add_parametric_RY_gate(1, np.pi/6)
        circuit.add_CNOT_gate(0, 1)
        prob = Probabilistic([p/4, p/4, p/4], [X(0), Y(0), Z(0)])
        circuit.add_gate(prob)

    noiseless_circuit = ParametricQuantumCircuit(n_qubit)
    noiseless_circuit.add_H_gate(0)
    noiseless_circuit.add_parametric_RY_gate(1, np.pi/6)
    noiseless_circuit.add_CNOT_gate(0, 1)

    n_sample_per_circuit = 1
    n_circuit_sample = 1000
    obs = Observable(n_qubit)
    obs.add_operator(1.0, "Z 0 Z 1")
    obs.add_operator(0.5, "X 0 X 1")
    initial_param = np.array([np.pi/6])

    opt = VqeOptimizer(parametric_circuit_list,
                       initial_state,
                       obs,
                       initial_param,
                       p_list,
                       n_circuit_sample=n_circuit_sample,
                       n_sample_per_circuit=n_sample_per_circuit,
                       noiseless_circuit=noiseless_circuit
                       )

    noisy = opt.sample_output(initial_param)
    mitigated, exp_array, _ = opt.sample_mitigated_output(
        initial_param, return_full=True)
    exact = opt.exact_output(initial_param)

    print(noisy, exact)
    print(exp_array, mitigated)

    opt_param = opt.optimize()
    print(opt_param)
    theta_list = np.linspace(0, np.pi, 100)
    output_list = [opt.exact_output([theta]) for theta in theta_list]
    plt.plot(theta_list, output_list, color="black",
             linestyle="dashed", label="exact")
    plt.scatter(opt.parameter_history, opt.exp_history,
                c="blue", label="optimization history")
    plt.xlabel("theta")
    plt.ylabel("output")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_VqeOptimizer()

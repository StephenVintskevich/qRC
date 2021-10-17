import numpy as np
from scipy.stats import entropy
from tqdm.notebook import tqdm
import qutip as qt
from qutip_qip.operations import cnot, hadamard_transform
import sys
from tqdm import tqdm

path_main = '/Users/stepanvinckevich/Desktop/IMPORTANT NOW/QIS QRL/CODE/QRC/'
sys.path.append(path_main)

'''SPECIFYING NUMBER OF QUBITS'''
n_qubitsA = 3
n_qubitsB = 2
# Evolution timestep
timestep = 2
# Trace preserving tolerance
tp_tol = 1e-5
n_qubits_total = n_qubitsA + n_qubitsB

'''UPLOADING HAMILTONIANS'''
# Hamiltonian -- A
hamiltonianA = qt.Qobj(np.load(path_main + '/hamiltonians/hamiltonian3.npy'), dims=[[2] * n_qubitsA, [2] * n_qubitsA])
identityA = qt.identity(2 ** n_qubitsA)
identityA.dims = [[2] * n_qubitsA, [2] * n_qubitsA]
# Hamiltonian -- B
hamiltonianB = qt.Qobj(np.load(path_main + '/hamiltonians/hamiltonian2.npy'), dims=[[2] * n_qubitsB, [2] * n_qubitsB])
identityB = qt.identity(2 ** n_qubitsB)
identityB.dims = [[2] * n_qubitsB, [2] * n_qubitsB]
# Full two-partite hamiltonian
hamiltonian = qt.tensor(hamiltonianA, identityB) + qt.tensor(identityA, hamiltonianB)

# Evolution step propagator
propagator = qt.propagator(hamiltonian, timestep)
propagator_dag = propagator.dag().full()
propagator = propagator.full()

# Evolution of entanglement sites a and b (now taken as identity)
prop_a = np.eye(2)
prop_b = np.eye(2)

# Operations
cnot = cnot(2, control=0, target=1).full()
hadamard = hadamard_transform().full()

# Entanglement channel
ent_transform = cnot @ (np.kron(hadamard @ prop_a, prop_b))
ent_transform_dag = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(ent_transform.T.conj(), np.eye(2 ** (n_qubitsB - 1))))
ent_transform = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(ent_transform, np.eye(2 ** (n_qubitsB - 1))))

# Initial density matrix
init_state = qt.tensor(*[qt.rand_ket(2) for _ in range(n_qubits_total)])
#init_state = init_state * init_state.dag()
init_state_qt =init_state * init_state.dag()
init_state = init_state_qt.full()

# np.save(f"TF/init_state{n_qubits}.npy", init_state)
# init_state = np.load(f"TF/init_state{n_qubits}.npy")

# Z observable and I/4
local_obs = np.array([[1., 0.], [0., -1.]])
identity_ent = np.eye(4) / 4


def ptrace(mtx, hold):
    """Performs partial trace of state with only one site #hold remains"""
    indexes = list(range(2 * n_qubits_total))
    axis_order = [indexes.pop(hold), indexes.pop(hold + n_qubits_total - 1)]
    axis_order = indexes + axis_order
    mtx = np.transpose(np.reshape(mtx, [2] * 2 * n_qubits_total), axis_order)
    return np.trace(np.reshape(mtx, (2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1), 2, 2)))


def evolution_step(state):
    """Simple application of evolution step propagator"""
    state = propagator @ state @ propagator_dag
    # assert np.abs(np.trace(state) - 1.) < tp_tol, f"Trace is not preserved.{np.trace(state)}"
    return state / np.trace(state)


def separated_measurement_step(state, site):
    """Nakajima pipeline"""
    # Qubit #site dm and Z observable application
    one_qubit_dm = ptrace(state, site)
    outcome = np.real(np.trace(local_obs @ one_qubit_dm))
    return state, outcome


def entangled_channel_step(state, p):
    """Application of entanglement channel with and depolarization channel with probability of depolarization 'p'"""
    state = ent_transform @ state @ ent_transform_dag
    if p != 0:
        # State dm mixed with state, where 'a + b' subsystem is replaced with identity matrix, with weights (1 - p) and (p) respectively
        axis_order = [n_qubitsA - 1, n_qubitsA, n_qubits_total + n_qubitsA - 1, n_qubits_total + n_qubitsA] + list(
            range(0, n_qubitsA - 1)) + list(range(n_qubitsA + 1, n_qubits_total + n_qubitsA - 1)) + list(
            range(n_qubits_total + n_qubitsA + 1, 2 * n_qubits_total))
        traced_dm = np.trace(np.reshape(
            np.transpose(np.reshape(state, [2] * 2 * n_qubits_total), axis_order),
            (4, 4, 2 ** (n_qubits_total - 2), 2 ** (n_qubits_total - 2))
        ))
        identity_inserted = np.kron(identity_ent, traced_dm)
        axis_order = list(range(2, n_qubitsA + 1)) + [0, 1] + list(range(n_qubitsA + 1, n_qubits_total)) + list(
            range(n_qubits_total + 2, n_qubits_total + n_qubitsA + 1)) + [n_qubits_total, n_qubits_total + 1] + list(
            range(n_qubits_total + n_qubitsA + 1, 2 * n_qubits_total))
        identity_inserted = np.reshape(
            np.transpose(np.reshape(state, [2] * 2 * n_qubits_total), axis_order),
            (2 ** n_qubits_total, 2 ** n_qubits_total)
        )
        state = p * identity_inserted + (1 - p) * state
    return state


def encoding_step(state, inp):
    """Encoding of inp to reservoir site #0"""
    # Partial trace. #0 site is traced out
    axis_order = [0, n_qubits_total] + list(range(1, n_qubits_total)) + list(
        range(n_qubits_total + 1, 2 * n_qubits_total))
    traced_dm = np.trace(np.reshape(
        np.transpose(np.reshape(state, [2] * 2 * n_qubits_total), axis_order),
        (2, 2, 2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1))
    ))
    # Encoding
    encoding_node_state = np.array([[np.sqrt(inp), np.sqrt(1. - inp)]])
    return np.kron(encoding_node_state.T @ encoding_node_state, traced_dm)


def get_mutual_information(state):
    axis_order = list(range(n_qubitsA, n_qubits_total)) + list(
        range(n_qubitsA + n_qubits_total, 2 * n_qubits_total)) + list(
        range(0, n_qubitsA)) + list(range(n_qubits_total, n_qubitsA + n_qubits_total))
    dmA = np.trace(np.reshape(
        np.transpose(np.reshape(state, [2] * 2 * n_qubits_total), axis_order),
        (2 ** n_qubitsB, 2 ** n_qubitsB, 2 ** n_qubitsA, 2 ** n_qubitsA)
    ))
    axis_order = list(range(0, n_qubitsA)) + list(range(n_qubits_total, n_qubitsA + n_qubits_total)) + list(
        range(n_qubitsA, n_qubits_total)) + list(range(n_qubitsA + n_qubits_total, 2 * n_qubits_total))
    dmB = np.trace(np.reshape(
        np.transpose(np.reshape(state, [2] * 2 * n_qubits_total), axis_order),
        (2 ** n_qubitsA, 2 ** n_qubitsA, 2 ** n_qubitsB, 2 ** n_qubitsB)
    ))
    eigsS, _ = np.linalg.eigh(state)
    eigsA, _ = np.linalg.eigh(dmA)
    eigsB, _ = np.linalg.eigh(dmB)
    return entropy(eigsS[eigsS > tp_tol]) - entropy(eigsA[eigsA > tp_tol]) - entropy(eigsB[eigsB > tp_tol])
    # return entropy(eigsS[eigsS > tp_tol]) - entropy(eigsA[eigsA > tp_tol]), entropy(eigsS[eigsS > tp_tol]) - entropy(eigsB[eigsB > tp_tol])


def get_Shmidt(state):
    axis_order = list(range(n_qubitsA, n_qubits_total)) + list(
        range(n_qubitsA + n_qubits_total, 2 * n_qubits_total)) + list(
        range(0, n_qubitsA)) + list(range(n_qubits_total, n_qubitsA + n_qubits_total))
    dmA = np.trace(np.reshape(
        np.transpose(np.reshape(state, [2] * 2 * n_qubits_total), axis_order),
        (2 ** n_qubitsB, 2 ** n_qubitsB, 2 ** n_qubitsA, 2 ** n_qubitsA)
    ))
    eigsA, _ = np.linalg.eigh(dmA)
    return np.sum(eigsA ** 2)


multiplexing = 30


# offset = 3 #( OFFSET CONTROLLED MANUALLY 1 -- 10)


def evaluate(depolarization_prob):
    # Initialization
    result = np.zeros((len(inp), multiplexing, n_qubitsA - 1))
    mutual_information = np.zeros((len(inp), multiplexing, n_qubitsA - 1))
    state = init_state
    # # Cook before processing
    # for _ in range(2000):
    #     state = evolution_step(state)

    for n in tqdm(range(len(inp))):
        # Encoding
        state = encoding_step(state, inp[n])
        # Main part
        for m in range(multiplexing):
            # On each step the only one site is "measured"
            for site in range(n_qubitsA - 1):
                # Evolution -> Entanglement -> Evolution
                mutual_information[n, m, site] = get_mutual_information(state)
                state = evolution_step(state)
                state = entangled_channel_step(state, depolarization_prob)
                state = evolution_step(state)

                # "Measurement"
                state, outcome = separated_measurement_step(state, site + 1)
                result[n, m, site] = outcome

    return result, mutual_information


def evaluate_no_enc(depolarization_prob):
    # Initialization
    mutual_information = np.zeros((1000, n_qubitsA - 1))
    state = init_state

    # Encoding
    state = encoding_step(state, 0.5)
    # Main part
    for m in tqdm(range(1000)):
        # On each step the only one site is "measured"
        for site in range(n_qubitsA - 1):
            # Evolution -> Entanglement -> Evolution
            mutual_information[m, site] = get_mutual_information(state)
            state = evolution_step(state)
            state = entangled_channel_step(state, depolarization_prob)
            state = evolution_step(state)

    return mutual_information


################### EVALUATION PROCEDURE #################################

# signal = np.load(path_main + '/data/complex_signal_9_9.npy')
# for offset in range(0, 10):
#     inp = signal[offset:]
#     target = signal[:-offset]
#     for p in np.linspace(0, 0.5, 51):
#         # Execution
#         print(p)
#         results, inf = evaluate(p)
#
#         np.save(
#             f"/Users/stepanvinckevich/Desktop/IMPORTANT NOW/QIS QRL/CODE/QRC/results/InfCapacity/STMOffest{offset}Results{p}.npy",
#             results)
#         np.save(
#             f"/Users/stepanvinckevich/Desktop/IMPORTANT NOW/QIS QRL/CODE/QRC/results/InfCapacity/STMOffest{offset}Information{p}.npy",
#             inf)



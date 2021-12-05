import numpy as np
from scipy.stats import entropy
import sys
from tqdm.notebook import tqdm
import qutip as qt
from qutip_qip.operations import cnot, hadamard_transform
import scipy as sc

path_main = '/Users/stepanvinckevich/Desktop/IMPORTANT NOW/QIS QRL/CODE/qRC/'

#### SET RESRVOIRS PARAMS ##############
# Number of qubits in reservoirs A and B
n_qubitsA = 3
k_power_A = 0

n_qubitsB = 3
k_power_B = 0

# Evolution timestep
timestep = 0.3
# Trace preserving tolerance
tp_tol = 1e-5

n_qubits_total = n_qubitsA + n_qubitsB

#### SET HAMILTONIANS ##############
# Hamiltonian of A
if k_power_A != 0:
    hamiltonianA = qt.Qobj(np.load(f'/content/gdrive/My Drive/QRC/hamiltonians/hamiltonian{n_qubitsA}_{k_power_A}.npy'),
                           dims=[[2] * n_qubitsA, [2] * n_qubitsA])
else:
    hamiltonianA = qt.Qobj(np.load(path_main + f'/hamiltonians/hamiltonian{n_qubitsA}.npy'),
                           dims=[[2] * n_qubitsA, [2] * n_qubitsA])
identityA = qt.identity(2 ** n_qubitsA)
identityA.dims = [[2] * n_qubitsA, [2] * n_qubitsA]

# Hamiltonian of B
if k_power_B != 0:
    hamiltonianB = qt.Qobj(np.load(f'/content/gdrive/My Drive/QRC/hamiltonians/hamiltonian{n_qubitsB}_{k_power_B}.npy'),
                           dims=[[2] * n_qubitsB, [2] * n_qubitsB])
else:
    hamiltonianB = qt.Qobj(np.load(path_main + f'/hamiltonians/hamiltonian{n_qubitsB}.npy'),
                           dims=[[2] * n_qubitsB, [2] * n_qubitsB])
identityB = qt.identity(2 ** n_qubitsB)
identityB.dims = [[2] * n_qubitsB, [2] * n_qubitsB]

# Total hamiltonian
hamiltonian = qt.tensor(hamiltonianA, identityB) + qt.tensor(identityA, hamiltonianB)
# hamiltonian = hamiltonianA
#### CAST UNITARY EVOLUTION AND MEASUREMENTS OPERATORS INTO NUMPY FORMAT


# Evolution step propagator
propagator = qt.propagator(hamiltonian, timestep)
propagator_dag = propagator.dag().full()
propagator = propagator.full()

# Evolution of entanglement sites a and b (now taken as identity)
prop_a = np.eye(2)
prop_b = np.eye(2)

# Operations
cnot_mtx = cnot(2, control=0, target=1).full()
hadamard = hadamard_transform().full()
cnot_mtx_backward = cnot(2, control=1, target=0).full()

# Entanglement channel
# ent_transform = cnot_mtx @ (np.kron(hadamard @ prop_a, prop_b))
ent_transform = cnot_mtx @ cnot_mtx_backward @ cnot_mtx @ (np.kron(hadamard @ prop_a, prop_b))

ent_transform_dag = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(ent_transform.T.conj(), np.eye(2 ** (n_qubitsB - 1))))
ent_transform = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(ent_transform, np.eye(2 ** (n_qubitsB - 1))))

#################################################

b = np.random.uniform(0, 1 / 3)
c = np.random.uniform(0, 1 / 3)

P0 = qt.basis(2, 0)
P1 = qt.basis(2, 1)

P00 = qt.tensor(P0, P0)
P01 = qt.tensor(P0, P1)
P10 = qt.tensor(P1, P0)
P11 = qt.tensor(P1, P1)

##### SET THE MAIN BASIS

psi00 = b * P00 + np.sqrt(1 - b ** 2) * P11
B00 = psi00 * psi00.dag()
B00 = B00.full()

psi11 = -b * P11 + np.sqrt(1 - b ** 2) * P00
B11 = psi11 * psi11.dag()
B11 = B11.full()

psi01 = c * P01 + np.sqrt(1 - c ** 2) * P10
B01 = psi01 * psi01.dag()
B01 = B01.full()

psi10 = -c * P10 + np.sqrt(1 - c ** 2) * P01
B10 = psi10 * psi10.dag()
B10 = B10.full()

PB00_dag = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(B00.T.conj(), np.eye(2 ** (n_qubitsB - 1))))
PB00 = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(B00, np.eye(2 ** (n_qubitsB - 1))))

PB11_dag = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(B11.T.conj(), np.eye(2 ** (n_qubitsB - 1))))
PB11 = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(B11, np.eye(2 ** (n_qubitsB - 1))))

PB01_dag = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(B01.T.conj(), np.eye(2 ** (n_qubitsB - 1))))
PB01 = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(B01, np.eye(2 ** (n_qubitsB - 1))))

PB10_dag = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(B10.T.conj(), np.eye(2 ** (n_qubitsB - 1))))
PB10 = np.kron(np.eye(2 ** (n_qubitsA - 1)), np.kron(B10, np.eye(2 ** (n_qubitsB - 1))))

######################

# Initial density matrix
init_state = qt.tensor(*[qt.rand_ket(2) for _ in range(n_qubits_total)])
init_state = init_state * init_state.dag()
init_state = init_state.full()

# Z observable and I/2
local_obs = np.array([[1., 0.], [0., -1.]])
identity_ent = np.eye(2) / 2
identity_total = np.eye(2 ** n_qubits_total)


####################### CORE. FUNCTIONS #############################

#
def get_depol_ops(site):
    sx = np.kron(np.eye(2 ** site), np.kron(np.array([[0., 1.], [1., 0.]]), np.eye(2 ** (n_qubits_total - site - 1))))
    sy = np.kron(np.eye(2 ** site), np.kron(np.array([[0., -1j], [1j, 0.]]), np.eye(2 ** (n_qubits_total - site - 1))))
    sz = np.kron(np.eye(2 ** site), np.kron(np.array([[1., 0.], [0., -1.]]), np.eye(2 ** (n_qubits_total - site - 1))))
    return sx, sy, sz


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


def evolution_step_obs(obs):
    """Simple application of evolution step propagator"""
    obs = propagator_dag @ obs @ propagator
    return obs


depol_ops = [get_depol_ops(site) for site in range(n_qubits_total)]


# FOR TESTING NONSELECTIVE MEASUREMENTS DONT FORGET TO CHANGE PARAMETER rand !!!!!

def entangled_channel_step(state, p, rand=True):
    """Application of entanglement channel with and depolarization channel with probability of depolarization 'p'"""
    if not rand:
        state = ent_transform @ state @ ent_transform_dag
    else:
        state = PB00 @ state @ PB00_dag + PB11 @ state @ PB11_dag + PB10 @ state @ PB10_dag + PB01 @ state @ PB01_dag

    if p != 0:
        for site in [0, 1, 4, 5]:
            state = (1 - p) * state + p / 3 * sum([depol_ops[site][i] @ state @ depol_ops[site][i] for i in range(3)])
    return state / np.trace(state)


def entangled_channel_step_obs(state, p, rand=True):
    """Application of entanglement channel with and depolarization channel with probability of depolarization 'p'"""
    if not rand:
        state = ent_transform_dag @ state @ ent_transform
    else:
        state = PB00_dag @ state @ PB00 + PB11_dag @ state @ PB11 + PB10_dag @ state @ PB10 + PB01_dag @ state @ PB01
    if p != 0:
        for site in [0, 1, 4, 5]:
            state = (1 - p) * state + p / 3 * sum([depol_ops[site][i] @ state @ depol_ops[site][i] for i in range(3)])
    return state


tracing_krauss0 = np.concatenate(
    [np.eye(2 ** (n_qubits_total - 1)), np.zeros((2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1)))], axis=1)
tracing_krauss1 = np.concatenate(
    [np.zeros((2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1))), np.eye(2 ** (n_qubits_total - 1))], axis=1)


def get_encoding_krauss(inp):
    tensor_product_krauss = np.eye(2 ** (n_qubits_total - 1)).ravel()
    tensor_product_krauss = np.stack(
        [tensor_product_krauss * np.sqrt(inp), tensor_product_krauss * np.sqrt(1. - inp)]).reshape(
        2 ** (n_qubits_total), 2 ** (n_qubits_total - 1))
    return tensor_product_krauss


def encoding_step(state, inp):
    """Encoding of inp to reservoir site #0"""
    # Partial trace. #0 site is traced out
    #     axis_order = [0, n_qubits_total] + list(range(1, n_qubits_total)) + list(
    #         range(n_qubits_total + 1, 2*n_qubits_total))
    #     traced_dm = np.trace(np.reshape(
    #         np.transpose(np.reshape(state, [2]*2*n_qubits_total), axis_order),
    #         (2, 2, 2**(n_qubits_total-1), 2**(n_qubits_total-1))
    #     ))
    #     # Encoding
    #     encoding_node_state = np.array([[np.sqrt(inp), np.sqrt(1. - inp)]])
    #     return np.kron(encoding_node_state.T @ encoding_node_state, traced_dm)
    enc_krauss = get_encoding_krauss(inp)
    state = tracing_krauss0 @ state @ tracing_krauss0.T + tracing_krauss1 @ state @ tracing_krauss1.T
    state = enc_krauss @ state @ enc_krauss.T
    return state


def separated_measurement_step(state, site):
    """Nakajima pipeline"""
    # Qubit #site dm and Z observable application
    one_qubit_dm = ptrace(state, site)
    outcome = np.real(np.trace(local_obs @ one_qubit_dm))
    return state, outcome


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


state = evolution_step_obs(init_state)
indexes = list(range(2 * n_qubits_total))
axis_order = [indexes.pop(0), indexes.pop(n_qubits_total - 1)]
axis_order = axis_order + indexes
state1 = np.transpose(np.reshape(state, [2] * 2 * n_qubits_total), axis_order)
state1 = np.trace(np.reshape(state1, (2, 2, 2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1))))

state2 = tracing_krauss0 @ state @ tracing_krauss0.T + tracing_krauss1 @ state @ tracing_krauss1.T

assert np.linalg.norm(state2 - state1) < 1e-12, "Tracing ops failure in value"
assert state2.shape == (2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1)), "Tracing ops failure in shape"

inp = 0.2
tensor_product_krauss = get_encoding_krauss(inp)
state = evolution_step(init_state)
state1 = encoding_step(state, inp)
state2 = tensor_product_krauss @ (
            tracing_krauss0 @ state @ tracing_krauss0.T + tracing_krauss1 @ state @ tracing_krauss1.T) @ tensor_product_krauss.T
assert np.linalg.norm(state2 - state1) < 1e-12, "Tensor product ops failure in value"
assert state2.shape == (2 ** n_qubits_total, 2 ** n_qubits_total), "Tensor product ops failure in shape"
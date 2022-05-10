import numpy as np
import qutip as qt
from qutip_qip.operations import cnot, hadamard_transform
import scipy as sc


identity = qt.identity(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
cx_aq = cnot(2, control=0, target=1)
cx_qa = cnot(2, control=1, target=0)

Pauli_basis = [sx, sy, sz]


def ptrace(mtx, hold, n_qubits_total):
    """Performs partial trace of state with only one site #hold remains"""
    indexes = list(range(2 * n_qubits_total))
    axis_order = [indexes.pop(hold), indexes.pop(hold + n_qubits_total - 1)]
    axis_order = indexes + axis_order
    mtx = np.transpose(np.reshape(mtx, [2] * 2 * n_qubits_total), axis_order)
    return np.trace(np.reshape(mtx, (2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1), 2, 2)))


def separated_measurement_step(state, site, n_qubits_total, local_obs=np.array([[1., 0.], [0., -1.]])):
    """Nakajima pipeline"""
    # Qubit #site dm and Z observable application
    one_qubit_dm = ptrace(state, site, n_qubits_total)
    outcome = np.real(np.trace(local_obs @ one_qubit_dm))
    return state, outcome


def get_purity(state, n_qubits_total):
    purity = []
    for site in range(n_qubits_total):
        one_qubit_dm = ptrace(state, site, n_qubits_total)
        purity.append(np.real(np.trace(one_qubit_dm @ one_qubit_dm)))
    return np.array(purity)


def rotation_unitary(theta, ang1, ang2):
    nx = np.cos(ang1) * np.sin(ang2)
    ny = np.sin(ang1) * np.sin(ang2)
    nz = np.cos(ang2)
    return np.cos(theta/2) * identity - 1j * np.sin(theta/2) * (nx * sx + ny * sy + nz * sz)


def get_anzatz(rotation1, rotation2):
    ra1 = rotation_unitary(*rotation1)
    ra2 = rotation_unitary(*rotation2)
    return cx_qa * qt.tensor(ra2, identity) * cx_aq * qt.tensor(ra1, identity)


def get_complex_anzatz(alpha, rotations):
    R = []
    for rotation in rotations:
        R.append(rotation_unitary(*rotation))
    inner_unitary = sc.linalg.expm(-1j * sum([alpha[i] * np.kron(Pauli_basis[i], Pauli_basis[i]) for i in range(3)]))
    return qt.Qobj(np.kron(R[0], R[1]) @ inner_unitary @ np.kron(R[2], R[3]), dims=[[2] * 2, [2] * 2])


def get_swap_anzatz():
    return cnot(2, control=0, target=1) * cnot(2, control=1, target=0) * cnot(2, control=0, target=1)


def create_narma(n, seed):
    signal = np.zeros(len(seed))
    for k in range(1, len(seed)):
        signal[k] = 0.3*signal[k-1] + 1.5*seed[k-n+1]*seed[k] + 0.1
        if k >= n:
            signal[k] += 0.05*signal[k-1]*np.sum(signal[k-n:k-1])
        else:
            signal[k] += 0.05*signal[k-1]*np.sum(signal[:k-1])
    return signal


def get_depol_ops(site, n_qubits_total):
    sx_full = np.kron(np.eye(2 ** site), np.kron(np.array([[0., 1.], [1., 0.]]),
                                           np.eye(2 ** (n_qubits_total - site - 1))))
    sy_full = np.kron(np.eye(2 ** site), np.kron(np.array([[0., -1j], [1j, 0.]]),
                                           np.eye(2 ** (n_qubits_total - site - 1))))
    sz_full = np.kron(np.eye(2 ** site), np.kron(np.array([[1., 0.], [0., -1.]]),
                                           np.eye(2 ** (n_qubits_total - site - 1))))
    return sx_full, sy_full, sz_full


class Encoder:
    def __init__(self, n_qubits_total):
        self.n_qubits_total = n_qubits_total
        self.tracing_krauss0 = np.concatenate(
            [np.eye(2 ** (n_qubits_total - 1)), np.zeros((2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1)))],
            axis=1)
        self.tracing_krauss1 = np.concatenate(
            [np.zeros((2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1))), np.eye(2 ** (n_qubits_total - 1))],
            axis=1)

    def get_encoding_krauss(self, inp):
        tensor_product_krauss = np.eye(2 ** (self.n_qubits_total - 1)).ravel()
        tensor_product_krauss = np.stack(
            [tensor_product_krauss * np.sqrt(inp), tensor_product_krauss * np.sqrt(1. - inp)]).reshape(
            2 ** self.n_qubits_total, 2 ** (self.n_qubits_total - 1))
        return tensor_product_krauss

    def encoding_step(self, state, inp):
        """Encoding of inp to reservoir site #0"""
        enc_krauss = self.get_encoding_krauss(inp)
        state = self.tracing_krauss0 @ state @ self.tracing_krauss0.T + self.tracing_krauss1 @ state @ self.tracing_krauss1.T
        state = enc_krauss @ state @ enc_krauss.T
        return state


class Bridge:
    def __init__(self, n_qubitsA, n_qubitsB, bridge_anzatz=get_complex_anzatz):
        self.n_qubitsA = n_qubitsA
        self.n_qubitsB = n_qubitsB
        self.bridge_anzatz = bridge_anzatz
        P0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
        P1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
        self.identity1 = qt.identity(2 ** (n_qubitsA - 1))
        self.identity2 = qt.identity(2 ** (n_qubitsB - 1))
        self.P00 = qt.tensor(self.identity1, P0, P0, self.identity2).full()
        self.P01 = qt.tensor(self.identity1, P0, P1, self.identity2).full()
        self.P10 = qt.tensor(self.identity1, P1, P0, self.identity2).full()
        self.P11 = qt.tensor(self.identity1, P1, P1, self.identity2).full()

        self.n_qubits_total = n_qubitsA + n_qubitsB

        self.bridge_unitary = None
        self.bridge_unitary_dag = None

    def set_bridge_unitary(self, *args, **kwargs):
        bridge_unitary = self.bridge_anzatz(*args, **kwargs)
        bridge_unitary = qt.tensor(self.identity1, bridge_unitary, self.identity2)
        self.bridge_unitary = bridge_unitary.full()
        self.bridge_unitary_dag = bridge_unitary.dag().full()

    def entangled_channel_step(self, state):
        """Application of entanglement channel with and depolarization channel with probability of depolarization 'p'"""
        state = self.bridge_unitary_dag @ state @ self.bridge_unitary
        state = self.P00 @ state @ self.P00 + self.P11 @ state @ self.P11 + self.P10 @ state @ self.P10 + self.P01 @ state @ self.P01
        return state


class Purifier:
    def __init__(self, n_qubits_total, purification_anzatz=get_complex_anzatz):
        self.n_qubits_total = n_qubits_total
        self.purification_anzatz = purification_anzatz
        self.P0 = qt.basis(2, 0) * qt.basis(2, 0).dag()

        self.anzatz_tracing_krauss0 = np.concatenate(
            [np.eye(2 ** n_qubits_total), np.zeros((2 ** n_qubits_total, 2 ** n_qubits_total))], axis=1)

        self.anzatz_tracing_krauss1 = np.concatenate(
            [np.zeros((2 ** n_qubits_total, 2 ** n_qubits_total)), np.eye(2 ** n_qubits_total)], axis=1)
        self.anzatz_unitary_list = None
        self.anzatz_unitary_list_dag = None

    def set_anzatz_unitary_list(self, *args, **kwargs):
        anzatz_unitary = self.purification_anzatz(*args, **kwargs)
        id_n = qt.identity(2 ** (self.n_qubits_total - 1))
        id_n.dims = [[2] * (self.n_qubits_total - 1), [2] * (self.n_qubits_total - 1)]
        anzatz_unitary = qt.tensor(anzatz_unitary, id_n)
        self.anzatz_unitary_list = []
        self.anzatz_unitary_list_dag = []
        for site in range(self.n_qubits_total):
            axis_order = [0] + list(range(2, site + 2)) + [1] + list(range(site + 2, self.n_qubits_total + 1))
            anzatz_on_site = anzatz_unitary.permute(axis_order)
            self.anzatz_unitary_list.append(anzatz_on_site.full())
            self.anzatz_unitary_list_dag.append(anzatz_on_site.dag().full())

    def purification_step(self, state, site):
        state_out = np.kron(self.P0, state)
        state_out = (self.anzatz_unitary_list[site] @ state_out @ self.anzatz_unitary_list_dag[site])
        state_out = (self.anzatz_tracing_krauss0 @ state_out @ self.anzatz_tracing_krauss0.T +
                     self.anzatz_tracing_krauss1 @ state_out @ self.anzatz_tracing_krauss1.T)
        return state_out

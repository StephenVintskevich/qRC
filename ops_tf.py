import numpy as np
import qutip as qt
from qutip_qip.operations import cnot, hadamard_transform
import tensorflow as tf


identity = qt.identity(2)
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()
cx_aq = tf.constant(cnot(2, control=0, target=1).full())
cx_qa = tf.constant(cnot(2, control=1, target=0).full())

Pauli_basis = [sx, sy, sz]


def ptrace(mtx, hold, n_qubits_total):
    """Performs partial trace of state with only one site #hold remains"""
    indexes = list(range(2*n_qubits_total))
    axis_order = [indexes.pop(hold), indexes.pop(hold + n_qubits_total - 1)]
    axis_order = axis_order + indexes
    mtx = tf.transpose(tf.reshape(mtx, [2]*2*n_qubits_total), axis_order)
    return tf.linalg.trace(tf.reshape(mtx, (2, 2, 2**(n_qubits_total-1), 2**(n_qubits_total-1))))


def tf_kron(x, y):
    return tf.linalg.LinearOperatorKronecker([tf.linalg.LinearOperatorFullMatrix(x), tf.linalg.LinearOperatorFullMatrix(y)]).to_dense()


def separated_measurement_step(state, site, n_qubits_total,
                               local_obs=tf.constant(np.array([[1., 0.], [0., -1.]]), dtype=tf.complex128)):
    """Nakajima pipeline"""
    # Qubit #site dm and Z observable application
    one_qubit_dm = ptrace(state, site, n_qubits_total)
    outcome = tf.math.real(tf.linalg.trace(local_obs @ one_qubit_dm))
    return state, outcome


def get_purity(state, n_qubits_total):
    purity = []
    for site in range(n_qubits_total):
        one_qubit_dm = ptrace(state, site, n_qubits_total)
        purity.append(np.real(np.trace(one_qubit_dm @ one_qubit_dm)))
    return np.array(purity)


def rotation_unitary(theta, ang1, ang2):
    nx = tf.cast(tf.math.cos(ang1), dtype=tf.complex128) * tf.cast(tf.math.sin(ang2), dtype=tf.complex128)
    ny = tf.cast(tf.math.sin(ang1), dtype=tf.complex128) * tf.cast(tf.math.sin(ang2), dtype=tf.complex128)
    nz = tf.cast(tf.math.cos(ang2), dtype=tf.complex128)
    return tf.cast(tf.math.cos(theta/2), dtype=tf.complex128) * identity - 1j * tf.cast(tf.math.sin(theta/2), dtype=tf.complex128) * (nx * sx + ny * sy + nz * sz)


def get_anzatz(rotation1, rotation2):
    ra1 = rotation_unitary(*rotation1)
    ra2 = rotation_unitary(*rotation2)
    return cx_qa @ tf_kron(ra2, identity) @ cx_aq @ (tf_kron(identity, ra1))


def get_complex_anzatz(alpha, rotations):
    R = []
    for rotation in rotations:
        R.append(rotation_unitary(*rotation))
    inner_unitary = tf.linalg.expm(-1j * sum([tf.cast(alpha[i], dtype=tf.complex128) * tf_kron(Pauli_basis[i], Pauli_basis[i]) for i in range(3)]))
    return tf_kron(R[0], R[1]) @ inner_unitary @ tf_kron(R[2], R[3])


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
    sx_full = tf.constant(
        np.kron(np.eye(2 ** site), np.kron(np.array([[0., 1.], [1., 0.]]),
                                           np.eye(2 ** (n_qubits_total - site - 1)))),
        dtype=tf.complex128
    )
    sy_full = tf.constant(
        np.kron(np.eye(2 ** site), np.kron(np.array([[0., -1j], [1j, 0.]]),
                                           np.eye(2 ** (n_qubits_total - site - 1)))),
        dtype=tf.complex128
    )
    sz_full = tf.constant(
        np.kron(np.eye(2 ** site), np.kron(np.array([[1., 0.], [0., -1.]]),
                                           np.eye(2 ** (n_qubits_total - site - 1)))),
        dtype=tf.complex128
    )
    return sx_full, sy_full, sz_full


class Encoder:
    def __init__(self, n_qubits_total):
        self.n_qubits_total = n_qubits_total
        self.tracing_krauss0 = tf.constant(np.concatenate(
            [np.eye(2 ** (n_qubits_total - 1)), np.zeros((2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1)))],
            axis=1), dtype=tf.complex128)
        self.tracing_krauss1 = tf.constant(np.concatenate(
            [np.zeros((2 ** (n_qubits_total - 1), 2 ** (n_qubits_total - 1))), np.eye(2 ** (n_qubits_total - 1))],
            axis=1), dtype=tf.complex128)

    def get_encoding_krauss(self, inp):
        tensor_product_krauss = np.eye(2 ** (self.n_qubits_total - 1)).ravel()
        tensor_product_krauss = np.stack(
            [tensor_product_krauss * np.sqrt(inp), tensor_product_krauss * np.sqrt(1. - inp)]).reshape(
            2 ** self.n_qubits_total, 2 ** (self.n_qubits_total - 1))
        return tf.constant(tensor_product_krauss, dtype=tf.complex128)

    def encoding_step(self, state, inp):
        """Encoding of inp to reservoir site #0"""
        enc_krauss = self.get_encoding_krauss(inp)
        state = self.tracing_krauss0 @ state @ tf.transpose(self.tracing_krauss0) + self.tracing_krauss1 @ state @ tf.transpose(
            self.tracing_krauss1)
        state = enc_krauss @ state @ tf.transpose(enc_krauss)
        return state


class Bridge:
    def __init__(self, n_qubitsA, n_qubitsB, bridge_anzatz=get_complex_anzatz):
        self.n_qubitsA = n_qubitsA
        self.n_qubitsB = n_qubitsB
        self.bridge_anzatz = bridge_anzatz
        P0 = qt.basis(2, 0) * qt.basis(2, 0).dag()
        P1 = qt.basis(2, 1) * qt.basis(2, 1).dag()
        identity1 = qt.identity(2 ** (n_qubitsA - 1))
        identity2 = qt.identity(2 ** (n_qubitsB - 1))
        self.P00 = tf.constant(qt.tensor(identity1, P0, P0, identity2).full(), dtype=tf.complex128)
        self.P01 = tf.constant(qt.tensor(identity1, P0, P1, identity2).full(), dtype=tf.complex128)
        self.P10 = tf.constant(qt.tensor(identity1, P1, P0, identity2).full(), dtype=tf.complex128)
        self.P11 = tf.constant(qt.tensor(identity1, P1, P1, identity2).full(), dtype=tf.complex128)

        self.identity1 = identity1.full()
        self.identity2 = identity2.full()

        self.n_qubits_total = n_qubitsA + n_qubitsB

        self.bridge_unitary = None
        self.bridge_unitary_dag = None

    def set_bridge_unitary(self, *args, **kwargs):
        bridge_unitary = self.bridge_anzatz(*args, **kwargs)
        bridge_unitary = tf_kron(self.identity1, tf_kron(bridge_unitary, self.identity2))
        self.bridge_unitary = tf.cast(bridge_unitary, dtype=tf.complex128)
        self.bridge_unitary_dag = tf.math.conj(tf.transpose(self.bridge_unitary))

    def entangled_channel_step(self, state):
        """Application of entanglement channel with and depolarization channel with probability of depolarization 'p'"""
        state = self.bridge_unitary_dag @ state @ self.bridge_unitary
        state = self.P00 @ state @ self.P00 + self.P11 @ state @ self.P11 + self.P10 @ state @ self.P10 + self.P01 @ state @ self.P01
        return state


class Purifier:
    def __init__(self, n_qubits_total, purification_anzatz=get_complex_anzatz):
        self.n_qubits_total = n_qubits_total
        self.purification_anzatz = purification_anzatz
        self.P0 = tf.constant((qt.basis(2, 0) * qt.basis(2, 0).dag()).full())

        self.anzatz_tracing_krauss0 = tf.constant(np.concatenate(
            [np.eye(2 ** n_qubits_total), np.zeros((2 ** n_qubits_total, 2 ** n_qubits_total))], axis=1),
            dtype=tf.complex128)

        self.anzatz_tracing_krauss1 = tf.constant(np.concatenate(
            [np.zeros((2 ** n_qubits_total, 2 ** n_qubits_total)), np.eye(2 ** n_qubits_total)], axis=1),
            dtype=tf.complex128)
        self.anzatz_unitary_list = None
        self.anzatz_unitary_list_dag = None

    def set_anzatz_unitary_list(self, *args, **kwargs):
        anzatz_unitary = self.purification_anzatz(*args, **kwargs)
        id_n = qt.identity(2 ** (self.n_qubits_total - 1)).full()
        anzatz_unitary = tf.reshape(tf_kron(anzatz_unitary, id_n), [2] * 2 * (self.n_qubits_total + 1))
        self.anzatz_unitary_list = []
        self.anzatz_unitary_list_dag = []
        for site in range(self.n_qubits_total):
            axis_order = ([0] + list(range(2, site + 2)) + [1] + list(range(site + 2, self.n_qubits_total + 1)) +
                          [self.n_qubits_total + 1] + list(range(self.n_qubits_total + 3, self.n_qubits_total + site + 3)) +
                          [self.n_qubits_total + 2] + list(range(self.n_qubits_total + site + 3, 2 * self.n_qubits_total + 2)))
            anzatz_on_site = tf.cast(
                tf.reshape(tf.transpose(anzatz_unitary, perm=axis_order),
                           (2 ** (self.n_qubits_total + 1), 2 ** (self.n_qubits_total + 1))),
                dtype=tf.complex128
            )
            self.anzatz_unitary_list.append(anzatz_on_site)
            self.anzatz_unitary_list_dag.append(tf.math.conj(tf.transpose(anzatz_on_site)))

    def purification_step(self, state, site):
        state_out = tf_kron(self.P0, state)
        state_out = (self.anzatz_unitary_list[site] @ state_out @ self.anzatz_unitary_list_dag[site])
        state_out = self.anzatz_tracing_krauss0 @ state_out @ tf.transpose(
            self.anzatz_tracing_krauss0) + self.anzatz_tracing_krauss1 @ state_out @ tf.transpose(self.anzatz_tracing_krauss1)
        return state_out

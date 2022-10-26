from qrc.ops import *


ALPHA_PURITY = [0.4406405 , 1.1309928 , 0.44251928]
ROTATIONS_PURITY = [
         [1.5706338 , 0.25005922, 0.7473237 ],
         [3.1415927 , 0.20195979, 0.7275571 ],
         [1.4890401, 0.3383612, 0.662342 ],
         [1.0503925, 0.2508376, 0.7540839],
        ]

ALPHA_BRIDGE = [0.21563943, 0.4693331 , 1.0228151 ]
ROTATIONS_BRIDGE = [
         [1.0030993, 0.2866138, 1.003308 ],
         [0.88676465, 0.282303  , 0.9138089 ],
         [ 0.86939156, -0.2769101 ,  0.9120728 ],
         [ 0.99995124, -0.28283787,  1.0098064 ],
        ]


class Pipeline:
    def __init__(self, n_qubitsA, n_qubitsB, timestep, multiplexing=10):
        self.n_qubitsA = n_qubitsA
        self.n_qubitsB = n_qubitsB
        self.multiplexing = multiplexing
        self.n_qubits_total = n_qubitsA + n_qubitsB
        init_state = qt.tensor(*[qt.rand_ket(2) for _ in range(self.n_qubits_total)])
        init_state = init_state * init_state.dag()
        self.init_state = init_state.full()
        self.encoder = Encoder(self.n_qubits_total)
        self.bridge = Bridge(n_qubitsA, n_qubitsB)
        self.purifier = Purifier(self.n_qubits_total)
        self.propagator, self.propagator_dag = self.get_propagator(timestep)
        self.depol_ops = [get_depol_ops(site, self.n_qubits_total) for site in range(self.n_qubits_total)]

    def get_propagator(self, timestep):
        hamiltonianA = qt.Qobj(np.load(f'../hamiltonians/hamiltonian{self.n_qubitsA}.npy'),
                               dims=[[2] * self.n_qubitsA, [2] * self.n_qubitsA])

        identityA = qt.identity(2 ** self.n_qubitsA)
        identityA.dims = [[2] * self.n_qubitsA, [2] * self.n_qubitsA]
        hamiltonianB = qt.Qobj(np.load(f'../hamiltonians/hamiltonian{self.n_qubitsB}.npy'),
                               dims=[[2] * self.n_qubitsB, [2] * self.n_qubitsB])

        identityB = qt.identity(2 ** self.n_qubitsB)
        identityB.dims = [[2] * self.n_qubitsB, [2] * self.n_qubitsB]
        hamiltonian = qt.tensor(hamiltonianA, identityB) + qt.tensor(identityA, hamiltonianB)
        propagator = qt.propagator(hamiltonian, timestep)
        propagator_dag = propagator.dag().full()
        propagator = propagator.full()

        return propagator, propagator_dag

    def get_purification_queue(self):
        while True:
            yield from [self.n_qubitsA - 1, self.n_qubitsA]

    def evolution_step(self, state):
        """Simple application of evolution step propagator"""
        state = self.propagator @ state @ self.propagator_dag
        # assert np.abs(np.trace(state) - 1.) < tp_tol, f"Trace is not preserved.{np.trace(state)}"
        return state

    def depolarization_step(self, state, p):
        if p != 0:
            for site in list(range(self.n_qubitsA - 1)) + list(range(self.n_qubitsA + 1, self.n_qubitsA + self.n_qubitsB)):
                state = (1 - p) * state + p / 3 * sum(
                    [self.depol_ops[site][i] @ state @ self.depol_ops[site][i] for i in range(3)])
        return state

    def evaluate(self, signal, alpha_bridge, rotations_bridge, alpha_purity, rotations_purity, depolarization_prob=0):
        # Initialization
        inp = signal[1000:1100]
        result = np.zeros((len(inp), self.multiplexing, self.n_qubits_total - 1))
        purification_queue = self.get_purification_queue()
        state = self.init_state
        self.bridge.set_bridge_unitary(alpha_bridge, rotations_bridge)
        self.purifier.set_anzatz_unitary_list(alpha_purity, rotations_purity)

        for i in range(1000):
            state = self.encoder.encoding_step(state, signal[i])
            state = self.evolution_step(state)
            state = self.bridge.entangled_channel_step(state)
            state = self.depolarization_step(state, depolarization_prob)
            if i % 2 == 0 and i != 0:
                state = self.evolution_step(state)
                site = next(purification_queue)
                state = self.purifier.purification_step(state, site)

        for n in range(len(inp)):
            state = self.encoder.encoding_step(state, inp[n])
            state = self.evolution_step(state)
            state = self.bridge.entangled_channel_step(state)
            state = self.depolarization_step(state, depolarization_prob)

            for m in range(self.multiplexing):
                state = self.evolution_step(state)
                if m % 5 == 0:
                    site = next(purification_queue)
                    state = self.purifier.purification_step(state, site)

                for site in range(self.n_qubits_total - 1):
                    state, outcome = separated_measurement_step(state, site + 1, self.n_qubits_total)
                    result[n, m, site] = outcome

        return result


class DefaultBridgePipeline(Pipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha_bridge = ALPHA_BRIDGE
        self.rotations_bridge = ROTATIONS_BRIDGE
        self.alpha_purity = ALPHA_PURITY
        self.rotations_purity = ROTATIONS_PURITY

    def evaluate(self, signal, depolarization_prob=0):
        inp = signal[1000:1200]
        result = np.zeros((len(inp), self.multiplexing, self.n_qubits_total))
        purification_queue = self.get_purification_queue()
        state = self.init_state
        self.bridge.set_bridge_unitary(self.alpha_bridge, self.rotations_bridge)
        self.purifier.set_anzatz_unitary_list(self.alpha_purity, self.rotations_purity)

        for i in range(1000):
            state = self.encoder.encoding_step(state, signal[i])
            state = self.evolution_step(state)
            state = self.bridge.entangled_channel_step(state)
            state = self.depolarization_step(state, depolarization_prob)
            if i % 2 == 0 and i != 0:
                state = self.evolution_step(state)
                site = next(purification_queue)
                state = self.purifier.purification_step(state, site)

        for n in range(len(inp)):
            state = self.encoder.encoding_step(state, inp[n])
            state = self.evolution_step(state)
            state = self.bridge.entangled_channel_step(state)
            state = self.depolarization_step(state, depolarization_prob)

            for m in range(self.multiplexing):
                state = self.evolution_step(state)
                if m % 5 == 0:
                    site = next(purification_queue)
                    state = self.purifier.purification_step(state, site)

                for site in range(self.n_qubits_total):
                    state, outcome = separated_measurement_step(state, site, self.n_qubits_total)
                    result[n, m, site] = outcome

        return result


class NoPurificationPipeline(DefaultBridgePipeline):
    def evaluate(self, signal, depolarization_prob=0):
        inp = signal[1000:1200]
        result = np.zeros((len(inp), self.multiplexing, self.n_qubits_total))
        state = self.init_state
        self.bridge.set_bridge_unitary(self.alpha_bridge, self.rotations_bridge)

        for i in range(1000):
            state = self.encoder.encoding_step(state, signal[i])
            state = self.evolution_step(state)
            state = self.bridge.entangled_channel_step(state)
            state = self.depolarization_step(state, depolarization_prob)

        for n in range(len(inp)):
            state = self.encoder.encoding_step(state, inp[n])
            state = self.evolution_step(state)
            state = self.bridge.entangled_channel_step(state)
            state = self.depolarization_step(state, depolarization_prob)

            for m in range(self.multiplexing):
                state = self.evolution_step(state)

                for site in range(self.n_qubits_total):
                    state, outcome = separated_measurement_step(state, site, self.n_qubits_total)
                    result[n, m, site] = outcome

        return result


class CorrelationMeasurementPipeline(DefaultBridgePipeline):
    def evaluate(self, signal, depolarization_prob=0):
        inp = signal[1000:1200]
        result = np.zeros((len(inp), self.multiplexing, self.n_qubits_total))
        state = self.init_state
        self.bridge.set_bridge_unitary(self.alpha_bridge, self.rotations_bridge)

        for i in range(1000):
            state = self.encoder.encoding_step(state, signal[i])
            state = self.evolution_step(state)
            state = self.bridge.entangled_channel_step(state)
            state = self.depolarization_step(state, depolarization_prob)

        for n in range(len(inp)):
            state = self.encoder.encoding_step(state, inp[n])
            state = self.evolution_step(state)
            state = self.bridge.entangled_channel_step(state)
            state = self.depolarization_step(state, depolarization_prob)

            for m in range(self.multiplexing):
                state = self.evolution_step(state)

                for site in range(self.n_qubits_total):
                    state, outcome = separated_measurement_step(state, site, self.n_qubits_total)
                    result[n, m, site] = outcome

        return result

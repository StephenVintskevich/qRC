
import numpy as np
from scipy.stats import entropy
from tqdm.notebook import tqdm
import qutip as qt
import QRC_3A2B_main
from QRC_3A2B_main import init_state_qt, get_mutual_information
from tqdm import tqdm

import sys
path_main = '/Users/stepanvinckevich/Desktop/IMPORTANT NOW/QIS QRL/CODE/QRC/'
sys.path.append(path_main)

# STEP_1 CHECK MUTUAL INFORMATION:
mi_init = get_mutual_information(init_state_qt)
purity_in = (init_state_qt * init_state_qt).tr()
print(purity_in)
if np.abs(purity_in - 1) < 1 - 10**-9 and mi_init == 0:
    print('Initial state of the both reservoirs is pure.\n State is factorized : mutual information is {}'.format(mi_init))
else:
    print('State is not pure. Warning: Analysis might be incomplete. Proceed with caution')


# STEP_2 STANDART ACCURACY VS MEAN INFORMATION:
# THIS TASK IS NOT READY!

offset = 10
loss = []
mean_inf = []
for p in np.linspace(0, 0.5, 51):
    print(p)
    results = np.load(f"/Users/stepanvinckevich/Desktop/IMPORTANT NOW/QIS QRL/CODE/QRC/results/STMTask/STMOffest{offset}Results{p}.npy")
    inf = np.load(f"/Users/stepanvinckevich/Desktop/IMPORTANT NOW/QIS QRL/CODE/QRC/results/STMTask/STMOffest{offset}Information{p}.npy")
    X = results.reshape(-1, (n_qubitsA-1) * multiplexing)
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)
    X_val = X[-1000:]
    X = X[1000:-1000]
    W = np.linalg.inv(X.T @ X + 10**(-12)*np.eye(X.shape[1])) @ X.T @ target[1000:-1000]
    res = X_val @ W
    loss.append(np.linalg.norm(res - target[-1000:]))
    mean_inf.append(np.mean(inf))

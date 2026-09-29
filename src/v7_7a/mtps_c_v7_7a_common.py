"""
================================================================================
MTPS-C v7.7-a — MÓDULO COMÚN
================================================================================
Funciones compartidas por los 4 stages de v7.7-a.
Importar al inicio de cada stage:
    from mtps_c_v7_7a_common import *

MÉTODO aux2b CONGELADO. v7.7-a NO entrena, NO ajusta. Solo varía N_ensemble.

CONSTANTES:
  k = 5, M = 25, eps_pred = 1e-4, eps_real = 1e-6, N_SPLITS = 5
  P_LEVELS = [0.30, 0.50, 0.80, 0.90, 0.95]
  N_ensemble varía por stage: {50, 100, 200}

DATASET MIXTO (60 configs):
  20 ID (seed=42) + 20 OOD cercano (seed=4242) + 20 OOD lejano (seed=7878)

SALIDAS por stage:
  partial_n{N}.pkl con dict {'principal': aggregated, 'control': aggregated, 'n': N}
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.decomposition import PCA
import time
import pickle
import warnings
warnings.filterwarnings("ignore")


# ==============================================================================
# CONSTANTES CONGELADAS DESDE v7.6-c
# ==============================================================================
G = 1.0
M_BODIES = 3
DIM = 3
STATE_DIM = M_BODIES * DIM * 2

K_HORIZON = 5.0
T_MIN = 2.0
T_MAX = 15.0

M_GROUND_TRUTHS = 25
EPS_REAL_DEFAULT = 1e-6
EPS_PRED_DEFAULT = 1e-4
N_SNAPSHOTS = 10
K_NEIGHBORS = 5
P_LEVELS = [0.30, 0.50, 0.80, 0.90, 0.95]
P_MAIN = 0.50
P_HIGH = 0.95
P_LEVEL_FOR_SIZE = 0.50
PCA_VAR_THRESHOLD = 0.95
N_GT_CALIB = 12
N_GT_EVAL = 13
N_SPLITS = 5

# Variable única de v7.7-a (cada stage usa uno)
N_ENSEMBLE_VALUES = [50, 100, 200]

# Umbrales C10
ERRCAL_THRESHOLD = 0.10
N_LEVELS_REQUIRED_C10A = 4
ERRCAL_THRESHOLD_C10B = 0.10
COV_HIGH_THRESHOLD = 0.90
SIZE_PROXY_MAX = 1.0
SIZE_PROXY_FRACTION_MIN = 0.80
STD_THRESHOLD_C10F = 0.05

# Dataset mixto
N_CONFIGS_PER_REGIMEN = 4
SEED_INDIST = 42
SEED_OOD_NEAR = 4242
SEED_OOD_FAR = 7878

REGIMENS_VIRIALIZED = ['figura_ocho', 'jerarquico', 'resonancia',
                       'euler_colineal', 'masas_asimetricas']
OOD_FAR_TYPES = ['separatriz', 'encuentro_violento', 'escape_temprano',
                  'masas_extremas', 'no_virializado']


# ==============================================================================
# GENERADORES IN-DISTRIBUTION (heredados v7.5-b, congelados)
# ==============================================================================

def initial_figure_eight(perturbation=0.0, rng=None):
    if rng is None: rng = np.random.default_rng()
    masses = np.ones(M_BODIES)
    r1 = np.array([0.97000436, -0.24308753, 0.0])
    r2 = -r1.copy()
    r3 = np.array([0.0, 0.0, 0.0])
    v3 = np.array([-0.93240737, -0.86473146, 0.0])
    v1 = -v3 / 2.0
    v2 = v1.copy()
    state = np.concatenate([r1, r2, r3, v1, v2, v3])
    if perturbation > 0:
        state += perturbation * rng.standard_normal(STATE_DIM)
    return masses, state


def initial_hierarchical(rng):
    masses = np.array([1.0, 1.0, 0.5])
    d_b = 0.3 + 0.1 * rng.random()
    r1 = np.array([d_b/2, 0.0, 0.0])
    r2 = np.array([-d_b/2, 0.0, 0.0])
    d_t = 3.0 + 1.0 * rng.random()
    angle = 2 * np.pi * rng.random()
    r3 = np.array([d_t * np.cos(angle), d_t * np.sin(angle), 0.0])
    v_orb = np.sqrt(G / d_b) / np.sqrt(2)
    v1 = np.array([0.0, v_orb, 0.0])
    v2 = np.array([0.0, -v_orb, 0.0])
    v_t = 0.3 * np.sqrt(G * 2.5 / d_t)
    v3 = np.array([-v_t * np.sin(angle), v_t * np.cos(angle), 0.0])
    return masses, np.concatenate([r1, r2, r3, v1, v2, v3])


def initial_resonance(rng):
    masses = np.array([1.0, 1.0, 0.3])
    d_b = 0.3
    r1 = np.array([d_b/2, 0.0, 0.0])
    r2 = np.array([-d_b/2, 0.0, 0.0])
    v_b = np.sqrt(G / d_b) / np.sqrt(2)
    v1 = np.array([0.0, v_b, 0.0])
    v2 = np.array([0.0, -v_b, 0.0])
    T_b = 2 * np.pi * d_b / (2 * v_b)
    ratio = 1.5 if rng.random() < 0.5 else 2.0
    T_t = ratio * T_b
    d_t = (G * 2.3 * T_t**2 / (4 * np.pi**2))**(1/3)
    angle = 2 * np.pi * rng.random()
    r3 = np.array([d_t * np.cos(angle), d_t * np.sin(angle), 0.0])
    v3m = np.sqrt(G * 2.3 / d_t)
    v3 = np.array([-v3m * np.sin(angle), v3m * np.cos(angle), 0.0])
    state = np.concatenate([r1, r2, r3, v1, v2, v3])
    return masses, state + 0.02 * rng.standard_normal(STATE_DIM)


def initial_euler_collinear(rng):
    masses = np.ones(M_BODIES)
    d = 1.0 + 0.1 * rng.standard_normal()
    r1 = np.array([-d, 0, 0])
    r2 = np.array([0, 0, 0])
    r3 = np.array([d, 0, 0])
    v_rot = 0.5 + 0.1 * rng.random()
    v1 = np.array([0, -v_rot, 0])
    v2 = np.array([0, 0, 0])
    v3 = np.array([0, v_rot, 0])
    state = np.concatenate([r1, r2, r3, v1, v2, v3])
    return masses, state + 0.01 * rng.standard_normal(STATE_DIM)


def initial_asymmetric_masses(rng):
    masses = np.array([10.0, 1.0, 1.0])
    r1 = np.array([0.0, 0.0, 0.0])
    d2 = 0.8 + 0.2 * rng.random()
    d3 = 1.5 + 0.3 * rng.random()
    a2 = 2 * np.pi * rng.random()
    a3 = a2 + np.pi + 0.3 * rng.standard_normal()
    r2 = np.array([d2 * np.cos(a2), d2 * np.sin(a2), 0.0])
    r3 = np.array([d3 * np.cos(a3), d3 * np.sin(a3), 0.0])
    v2m = np.sqrt(G * 10.0 / d2)
    v3m = np.sqrt(G * 10.0 / d3)
    v2 = np.array([-v2m * np.sin(a2), v2m * np.cos(a2), 0.0])
    v3 = np.array([-v3m * np.sin(a3), v3m * np.cos(a3), 0.0])
    v1 = -(masses[1] * v2 + masses[2] * v3) / masses[0]
    return masses, np.concatenate([r1, r2, r3, v1, v2, v3])


REGIME_GENERATORS = {
    'figura_ocho':       lambda rng: initial_figure_eight(0.05 * rng.random(), rng),
    'jerarquico':        initial_hierarchical,
    'resonancia':        initial_resonance,
    'euler_colineal':    initial_euler_collinear,
    'masas_asimetricas': initial_asymmetric_masses,
}


# ==============================================================================
# GENERADORES OOD LEJANO (heredados v7.6-b, congelados)
# ==============================================================================

def gen_separatriz(rng):
    masses = np.array([1.0, 1.0, 0.5])
    d_b = 0.4
    r1 = np.array([d_b/2, 0.0, 0.0])
    r2 = np.array([-d_b/2, 0.0, 0.0])
    v_b = np.sqrt(G / d_b) / np.sqrt(2)
    v1 = np.array([0.0, v_b, 0.0])
    v2 = np.array([0.0, -v_b, 0.0])
    d_t = 2.0 + 0.3 * rng.standard_normal()
    M_int = 2.0
    v_marginal = np.sqrt(2 * G * M_int / d_t)
    factor = 0.85 + 0.15 * rng.random()
    angle = 2 * np.pi * rng.random()
    r3 = np.array([d_t * np.cos(angle), d_t * np.sin(angle), 0.0])
    v3m = factor * v_marginal
    radial_frac = 0.3 * rng.standard_normal()
    v3 = np.array([
        -v3m * np.sin(angle) + radial_frac * np.cos(angle),
        v3m * np.cos(angle) + radial_frac * np.sin(angle),
        0.0
    ])
    state = np.concatenate([r1, r2, r3, v1, v2, v3])
    return masses, state + 0.005 * rng.standard_normal(STATE_DIM)


def gen_encuentro_violento(rng):
    masses = np.ones(M_BODIES)
    scale = 0.15 + 0.05 * rng.random()
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3]) + 0.2 * rng.standard_normal()
    positions = np.array([[scale * np.cos(a), scale * np.sin(a), 0.0]
                            for a in angles])
    cm = np.mean(positions, axis=0)
    velocities = np.zeros((3, 3))
    v_speed = 1.5 + 0.3 * rng.random()
    for i in range(3):
        toward_cm = cm - positions[i]
        toward_cm /= np.linalg.norm(toward_cm) + 1e-12
        tangent = np.array([-toward_cm[1], toward_cm[0], 0])
        velocities[i] = v_speed * (0.7 * toward_cm + 0.3 * tangent)
    state = np.concatenate([positions.flatten(), velocities.flatten()])
    return masses, state + 0.003 * rng.standard_normal(STATE_DIM)


def gen_escape_temprano(rng):
    masses = np.array([1.0, 1.0, 0.5])
    d_b = 0.5
    r1 = np.array([d_b/2, 0.0, 0.0])
    r2 = np.array([-d_b/2, 0.0, 0.0])
    v_b = np.sqrt(G / d_b) / np.sqrt(2)
    v1 = np.array([0.0, v_b, 0.0])
    v2 = np.array([0.0, -v_b, 0.0])
    d_t = 1.0 + 0.3 * rng.random()
    angle = 2 * np.pi * rng.random()
    r3 = np.array([d_t * np.cos(angle), d_t * np.sin(angle), 0.0])
    M_int = 2.0
    v_escape = np.sqrt(2 * G * M_int / d_t)
    factor = 1.05 + 0.20 * rng.random()
    v3 = factor * v_escape * np.array([np.cos(angle), np.sin(angle), 0.0])
    state = np.concatenate([r1, r2, r3, v1, v2, v3])
    return masses, state + 0.005 * rng.standard_normal(STATE_DIM)


def gen_masas_extremas(rng):
    if rng.random() < 0.5:
        masses = np.array([100.0, 1.0, 1.0])
    else:
        masses = np.array([50.0, 5.0, 1.0])
    M_main = masses[0]
    r1 = np.array([0.0, 0.0, 0.0])
    d2 = 1.0 + 0.3 * rng.random()
    d3 = 2.0 + 0.5 * rng.random()
    a2 = 2 * np.pi * rng.random()
    a3 = a2 + np.pi + 0.2 * rng.standard_normal()
    r2 = np.array([d2 * np.cos(a2), d2 * np.sin(a2), 0.0])
    r3 = np.array([d3 * np.cos(a3), d3 * np.sin(a3), 0.0])
    v2m = np.sqrt(G * M_main / d2)
    v3m = np.sqrt(G * M_main / d3)
    v2 = np.array([-v2m * np.sin(a2), v2m * np.cos(a2), 0.0])
    v3 = np.array([-v3m * np.sin(a3), v3m * np.cos(a3), 0.0])
    v1 = -(masses[1] * v2 + masses[2] * v3) / M_main
    state = np.concatenate([r1, r2, r3, v1, v2, v3])
    return masses, state + 0.003 * rng.standard_normal(STATE_DIM)


def gen_no_virializado(rng):
    masses = np.ones(M_BODIES)
    angles = np.array([0, 2*np.pi/3, 4*np.pi/3])
    scale = 1.0
    positions = np.array([[scale * np.cos(a), scale * np.sin(a), 0.0]
                            for a in angles])
    U = 0.0
    for i in range(M_BODIES):
        for j in range(i+1, M_BODIES):
            r = np.linalg.norm(positions[i] - positions[j]) + 1e-3
            U -= G * masses[i] * masses[j] / r
    if rng.random() < 0.5:
        factor = 0.2 + 0.3 * rng.random()
    else:
        factor = 1.5 + 0.5 * rng.random()
    T_target = factor * abs(U) / 2.0
    velocities_raw = rng.standard_normal((3, 3))
    velocities_raw[:, 2] *= 0.1
    KE_raw = 0.5 * np.sum(masses[:, None] * velocities_raw**2)
    velocities = velocities_raw * np.sqrt(T_target / KE_raw)
    p_total = np.sum(masses[:, None] * velocities, axis=0)
    velocities -= p_total / np.sum(masses)
    state = np.concatenate([positions.flatten(), velocities.flatten()])
    return masses, state + 0.003 * rng.standard_normal(STATE_DIM)


OOD_FAR_GENERATORS = {
    'separatriz': gen_separatriz,
    'encuentro_violento': gen_encuentro_violento,
    'escape_temprano': gen_escape_temprano,
    'masas_extremas': gen_masas_extremas,
    'no_virializado': gen_no_virializado,
}


def generate_mixed_dataset():
    """Genera dataset mixto: 20 ID + 20 OOD cercano + 20 OOD lejano = 60 configs."""
    dataset = []

    # IN-DISTRIBUTION (seed=42)
    rng_id = np.random.default_rng(seed=SEED_INDIST)
    for reg_idx, regime_name in enumerate(REGIMENS_VIRIALIZED):
        gen = REGIME_GENERATORS[regime_name]
        for i in range(N_CONFIGS_PER_REGIMEN):
            masses, state = gen(rng_id)
            dataset.append({
                'config_id': f"ID_{regime_name}_{i:02d}",
                'subset': 'in_distribution',
                'regime_or_subtype': regime_name,
                'masses': masses.tolist() if hasattr(masses, 'tolist') else list(masses),
                'state_0': state.tolist() if hasattr(state, 'tolist') else list(state),
                'source_seed': SEED_INDIST,
            })

    # OOD CERCANO (seed=4242)
    rng_near = np.random.default_rng(seed=SEED_OOD_NEAR)
    for reg_idx, regime_name in enumerate(REGIMENS_VIRIALIZED):
        gen = REGIME_GENERATORS[regime_name]
        for i in range(N_CONFIGS_PER_REGIMEN):
            masses, state = gen(rng_near)
            dataset.append({
                'config_id': f"OOD_NEAR_{regime_name}_{i:02d}",
                'subset': 'ood_near',
                'regime_or_subtype': regime_name,
                'masses': masses.tolist() if hasattr(masses, 'tolist') else list(masses),
                'state_0': state.tolist() if hasattr(state, 'tolist') else list(state),
                'source_seed': SEED_OOD_NEAR,
            })

    # OOD LEJANO (seed=7878)
    rng_far = np.random.default_rng(seed=SEED_OOD_FAR)
    for type_idx, ood_subtype in enumerate(OOD_FAR_TYPES):
        gen = OOD_FAR_GENERATORS[ood_subtype]
        for i in range(N_CONFIGS_PER_REGIMEN):
            masses, state = gen(rng_far)
            dataset.append({
                'config_id': f"OOD_FAR_{ood_subtype}_{i:02d}",
                'subset': 'ood_far',
                'regime_or_subtype': ood_subtype,
                'masses': masses.tolist() if hasattr(masses, 'tolist') else list(masses),
                'state_0': state.tolist() if hasattr(state, 'tolist') else list(state),
                'source_seed': SEED_OOD_FAR,
            })

    return dataset


# ==============================================================================
# DINÁMICA Y CONFORMAL (congelados)
# ==============================================================================

def unpack_state(state):
    return state[:9].reshape(M_BODIES, DIM), state[9:18].reshape(M_BODIES, DIM)


def gravitational_rhs(t, state, masses, softening=1e-3):
    positions, velocities = unpack_state(state)
    accelerations = np.zeros_like(positions)
    for i in range(M_BODIES):
        for j in range(M_BODIES):
            if i == j: continue
            r_ij = positions[j] - positions[i]
            dist2 = np.sum(r_ij**2) + softening**2
            accelerations[i] += G * masses[j] * r_ij / dist2**1.5
    return np.concatenate([velocities.flatten(), accelerations.flatten()])


def compute_T_characteristic(state_0, masses):
    positions, _ = unpack_state(state_0)
    min_period = np.inf
    for i in range(M_BODIES):
        for j in range(i+1, M_BODIES):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            if r_ij < 1e-4: continue
            m_pair = masses[i] + masses[j]
            T_pair = 2 * np.pi * np.sqrt(r_ij**3 / (G * m_pair))
            if T_pair < min_period: min_period = T_pair
    return min_period if np.isfinite(min_period) else 1.0


def compute_T_adaptive(state_0, masses):
    return np.clip(K_HORIZON * compute_T_characteristic(state_0, masses),
                   T_MIN, T_MAX)


def integrate_with_snapshots(masses, state_0, T_max, n_snapshots=N_SNAPSHOTS):
    t_eval = np.linspace(T_max / n_snapshots, T_max, n_snapshots)
    sol = solve_ivp(
        gravitational_rhs, (0.0, T_max), state_0, args=(masses,),
        method='RK45', rtol=1e-8, atol=1e-10, max_step=0.05,
        t_eval=t_eval
    )
    if not sol.success:
        return None, None
    return sol.y.T, sol.t


def perturb_state(state_0, epsilon, rng):
    return state_0 + epsilon * rng.standard_normal(STATE_DIM)


def compute_knn_score(query_point, reference_set, k=K_NEIGHBORS,
                       exclude_self=False, self_idx=None):
    distances = cdist(query_point.reshape(1, -1), reference_set)[0]
    if exclude_self and self_idx is not None:
        mask = np.ones(len(distances), dtype=bool)
        mask[self_idx] = False
        distances = distances[mask]
    distances_sorted = np.sort(distances)
    if len(distances_sorted) < k:
        return float('inf')
    return float(distances_sorted[k - 1])


def compute_conformal_threshold(scores_calibration, p):
    if len(scores_calibration) == 0:
        return float("inf")
    scores = np.sort(np.asarray(scores_calibration))
    n = len(scores)
    idx = int(np.ceil((n + 1) * p)) - 1
    idx = min(max(idx, 0), n - 1)
    return float(scores[idx])


def fit_pca(ensemble_k, var_threshold=PCA_VAR_THRESHOLD):
    pca_full = PCA(n_components=min(ensemble_k.shape[0]-1, ensemble_k.shape[1]))
    pca_full.fit(ensemble_k)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    d_eff_raw = int(np.searchsorted(cum_var, var_threshold) + 1)
    d_eff_raw = max(1, min(d_eff_raw, len(cum_var)))
    d_eff_model = max(2, d_eff_raw)
    d_eff_model = min(d_eff_model, ensemble_k.shape[0] - 1, ensemble_k.shape[1])
    pca = PCA(n_components=d_eff_model)
    ensemble_pca = pca.fit_transform(ensemble_k)
    return pca, ensemble_pca, d_eff_raw, d_eff_model


# ==============================================================================
# EVALUACIÓN POR CONFIG (parametrizada por N_ensemble)
# ==============================================================================

def evaluate_config_v77a(config, seed_ensemble, seed_gt_base, mode, n_ensemble):
    """Idéntica a v7.6-c salvo que N_ensemble es parámetro."""
    masses = np.array(config['masses'])
    state_0 = np.array(config['state_0'])
    T_max = compute_T_adaptive(state_0, masses)

    # Ensemble
    rng_pred = np.random.default_rng(seed=seed_ensemble)
    ensemble_trajs = []
    for k in range(n_ensemble):
        state_0_pred = perturb_state(state_0, EPS_PRED_DEFAULT, rng_pred)
        traj, _ = integrate_with_snapshots(masses, state_0_pred, T_max)
        if traj is None: continue
        ensemble_trajs.append(traj)
    if len(ensemble_trajs) < n_ensemble // 2:
        return {'config_id': config['config_id'], 'subset': config['subset'],
                'integration_ok': False, 'reason': 'ensemble_failed'}
    ensemble_trajs = np.array(ensemble_trajs)

    # GTs
    ground_truths = []
    for m_idx in range(M_GROUND_TRUTHS):
        rng_gt = np.random.default_rng(seed=seed_gt_base + m_idx)
        state_0_gt = perturb_state(state_0, EPS_REAL_DEFAULT, rng_gt)
        traj_gt, times = integrate_with_snapshots(masses, state_0_gt, T_max)
        if traj_gt is None: continue
        ground_truths.append(traj_gt)
    if len(ground_truths) < M_GROUND_TRUTHS // 2:
        return {'config_id': config['config_id'], 'subset': config['subset'],
                'integration_ok': False, 'reason': 'gt_failed'}
    ground_truths = np.array(ground_truths)
    n_gt_total = len(ground_truths)

    # Splits
    if mode == 'v75b_aux2b':
        rng_split_master = np.random.default_rng(seed=seed_gt_base + 99999)
        splits = []
        for s in range(N_SPLITS):
            indices = rng_split_master.permutation(n_gt_total)
            n_calib_actual = min(N_GT_CALIB, n_gt_total // 2)
            calib_idx = indices[:n_calib_actual]
            eval_idx = indices[n_calib_actual:]
            splits.append((calib_idx, eval_idx))
    else:
        splits = [(None, np.arange(n_gt_total))]

    snapshots_data = []
    for k in range(N_SNAPSHOTS):
        ensemble_k = ensemble_trajs[:, k, :]
        pca, ensemble_pca, d_eff_raw, d_eff_model = fit_pca(ensemble_k)

        cov_pca_mat = np.cov(ensemble_pca.T) + 1e-10 * np.eye(d_eff_model)
        eigvals_cov = np.linalg.eigvalsh(cov_pca_mat)
        chi2_thresh_env_pca = chi2.ppf(0.95, df=d_eff_model)
        radio_env_95 = float(np.sqrt(chi2_thresh_env_pca * np.max(eigvals_cov)))

        per_split_data = []
        for s_idx, (calib_idx, eval_idx) in enumerate(splits):
            if mode == 'v75a_control':
                scores_calib = np.array([
                    compute_knn_score(ensemble_pca[i], ensemble_pca,
                                        k=K_NEIGHBORS, exclude_self=True, self_idx=i)
                    for i in range(ensemble_pca.shape[0])
                ])
            else:
                gt_states_k = ground_truths[:, k, :]
                gt_pca_all = pca.transform(gt_states_k)
                scores_calib = np.array([
                    compute_knn_score(gt_pca_all[idx], ensemble_pca,
                                        k=K_NEIGHBORS, exclude_self=False)
                    for idx in calib_idx
                ])

            tau_p = {p: compute_conformal_threshold(scores_calib, p)
                      for p in P_LEVELS}

            if mode == 'v75b_aux2b':
                eval_gts_pca = gt_pca_all[eval_idx]
            else:
                eval_gts_pca = pca.transform(ground_truths[:, k, :])

            n_eval = len(eval_gts_pca)
            cov_per_p = {p: 0 for p in P_LEVELS}
            for i in range(n_eval):
                score_gt = compute_knn_score(eval_gts_pca[i], ensemble_pca,
                                               k=K_NEIGHBORS, exclude_self=False)
                for p in P_LEVELS:
                    if score_gt <= tau_p[p]:
                        cov_per_p[p] += 1
            cov_rates = {p: cov_per_p[p] / n_eval for p in P_LEVELS}

            if radio_env_95 > 0:
                ratio_lineal = tau_p[P_LEVEL_FOR_SIZE] / radio_env_95
                size_proxy = float(ratio_lineal ** d_eff_model)
            else:
                size_proxy = float('inf')

            per_split_data.append({
                'split_id': s_idx,
                'tau_p': tau_p,
                'cov_rates': cov_rates,
                'size_proxy': size_proxy,
                'n_calib': len(calib_idx) if calib_idx is not None else n_ensemble,
                'n_eval': n_eval,
            })

        cov_rates_mean = {p: np.mean([s['cov_rates'][p] for s in per_split_data])
                           for p in P_LEVELS}
        size_proxies_split = [s['size_proxy'] for s in per_split_data
                                if np.isfinite(s['size_proxy'])]
        size_proxy_mean = float(np.mean(size_proxies_split)) if size_proxies_split else float('inf')
        errcal_mean = {p: abs(cov_rates_mean[p] - p) for p in P_LEVELS}

        snapshots_data.append({
            'k': k, 't': times[k],
            'd_eff_raw': d_eff_raw, 'd_eff_model': d_eff_model,
            'cov_rates_mean': cov_rates_mean,
            'errcal_mean': errcal_mean,
            'size_proxy_mean': size_proxy_mean,
            'per_split': per_split_data,
        })

    return {
        'config_id': config['config_id'],
        'subset': config['subset'],
        'regime_or_subtype': config['regime_or_subtype'],
        'mode': mode,
        'n_ensemble': n_ensemble,
        'integration_ok': True,
        'snapshots': snapshots_data,
    }


# ==============================================================================
# AGREGACIÓN POR N
# ==============================================================================

def aggregate_results_by_n(all_results, mode_label, n_ensemble):
    """Agregación específica para un valor de N."""
    rows_long = []
    rows_per_cell = []
    n_failed = 0

    for r in all_results:
        if not r.get('integration_ok', False):
            n_failed += 1
            continue
        for snap in r['snapshots']:
            cell = {
                'mode': mode_label, 'n_ensemble': n_ensemble,
                'config_id': r['config_id'], 'subset': r['subset'],
                'regime_or_subtype': r['regime_or_subtype'],
                'snapshot': snap['k'] + 1, 't': snap['t'],
                'd_eff_raw': snap['d_eff_raw'], 'd_eff_model': snap['d_eff_model'],
                'size_proxy_mean': snap['size_proxy_mean'],
            }
            for p in P_LEVELS:
                cell[f'cov_{int(100*p)}_mean'] = snap['cov_rates_mean'][p]
                cell[f'errcal_{int(100*p)}_mean'] = snap['errcal_mean'][p]
            rows_per_cell.append(cell)

            for split_data in snap['per_split']:
                for p in P_LEVELS:
                    rows_long.append({
                        'n_ensemble': n_ensemble,
                        'mode': mode_label,
                        'subset': r['subset'],
                        'regime_or_subtype': r['regime_or_subtype'],
                        'config_id': r['config_id'],
                        'snapshot': snap['k'] + 1,
                        'split_id': split_data['split_id'],
                        'p_level': p,
                        'cov_emp': split_data['cov_rates'][p],
                        'errcal': abs(split_data['cov_rates'][p] - p),
                        'tau_p': split_data['tau_p'][p],
                        'size_proxy': split_data['size_proxy'],
                    })

    if not rows_per_cell:
        return None

    cov_global = {p: np.mean([r[f'cov_{int(100*p)}_mean'] for r in rows_per_cell])
                   for p in P_LEVELS}
    errcal_global = {p: abs(cov_global[p] - p) for p in P_LEVELS}
    n_levels_ok = sum(1 for p in P_LEVELS if errcal_global[p] <= ERRCAL_THRESHOLD)

    by_subset = {}
    for r in rows_per_cell:
        by_subset.setdefault(r['subset'], []).append(r)
    cov_by_subset = {}
    for subset in ['in_distribution', 'ood_near', 'ood_far']:
        rs = by_subset.get(subset, [])
        if not rs: continue
        cov_by_subset[subset] = {p: np.mean([r[f'cov_{int(100*p)}_mean'] for r in rs])
                                    for p in P_LEVELS}

    c10a = n_levels_ok >= N_LEVELS_REQUIRED_C10A
    c10b = errcal_global[P_MAIN] <= ERRCAL_THRESHOLD_C10B
    c10c = cov_global[P_HIGH] >= COV_HIGH_THRESHOLD
    proxies = [r['size_proxy_mean'] for r in rows_per_cell
                if np.isfinite(r['size_proxy_mean'])]
    frac_size_ok = float(np.mean(np.array(proxies) <= SIZE_PROXY_MAX)) if proxies else 0.0
    c10d = frac_size_ok >= SIZE_PROXY_FRACTION_MIN

    split_ids = sorted(set(r['split_id'] for r in rows_long))
    errcals_by_split = []
    for sid in split_ids:
        vals = [r['cov_emp'] for r in rows_long
                if r['split_id'] == sid and abs(r['p_level'] - P_MAIN) < 1e-12]
        if vals:
            errcals_by_split.append(abs(np.mean(vals) - P_MAIN))
    if len(errcals_by_split) > 1:
        split_std_errcal_50 = float(np.std(errcals_by_split, ddof=1))
    else:
        split_std_errcal_50 = 0.0
    c10f = split_std_errcal_50 <= STD_THRESHOLD_C10F

    return {
        'mode': mode_label, 'n_ensemble': n_ensemble,
        'rows_per_cell': rows_per_cell, 'rows_long': rows_long,
        'cov_global': cov_global, 'errcal_global': errcal_global,
        'n_levels_ok': n_levels_ok, 'cov_by_subset': cov_by_subset,
        'c10a': c10a, 'c10b': c10b, 'c10c': c10c, 'c10d': c10d, 'c10f': c10f,
        'frac_size_ok': frac_size_ok,
        'split_std_errcal_50': split_std_errcal_50,
        'n_failed': n_failed,
    }


# ==============================================================================
# CLASIFICACIÓN DE ESCENARIO (con microcorrección del auditor aplicada)
# ==============================================================================

def classify_v77a_scenario(results_principal):
    """Microcorrección del auditor aplicada (S10.1 exige no errcal_increasing,
    S10.3 exige std_fails AND std_diverging, caso INTERMEDIO específico)."""
    n_values = sorted(results_principal.keys())

    all_cumplen = all(
        results_principal[n]['c10a'] and results_principal[n]['c10b']
        and results_principal[n]['c10c'] and results_principal[n]['c10d']
        for n in n_values
    )

    stds = [results_principal[n]['split_std_errcal_50'] for n in n_values]
    std_ok_todos = all(s <= STD_THRESHOLD_C10F for s in stds)
    std_diverging = (len(stds) >= 2 and stds[-1] > stds[0] + 0.01)

    errcals = [results_principal[n]['errcal_global'][P_MAIN] for n in n_values]
    errcal_decreasing = (len(errcals) >= 2 and errcals[-1] < errcals[0] - 0.01)
    errcal_increasing = (len(errcals) >= 2 and errcals[-1] > errcals[0] + 0.01)

    falla_n_alto = (not all_cumplen) and (
        not (results_principal[n_values[-1]]['c10a']
             and results_principal[n_values[-1]]['c10b']
             and results_principal[n_values[-1]]['c10c']
             and results_principal[n_values[-1]]['c10d'])
    )

    std_fails = not std_ok_todos

    if all_cumplen and std_ok_todos and not errcal_increasing:
        scenario = "S10.1"
        veredicto = "ROBUSTEZ AL ESCALAMIENTO"
        next_step = ("El método es robusto al tamaño del ensemble en {50, 100, 200}.\n"
                     "Próximo paso: v7.7-b sensibilidad a hyperparámetros\n"
                     "(con N=50 baseline para mayor velocidad).")

    elif falla_n_alto and errcal_increasing:
        scenario = "S10.2"
        veredicto = "DEGRADACIÓN POR ESCALAMIENTO"
        next_step = ("El método se degrada con N creciente. Sobre-ajuste geométrico.\n"
                     "Posible v7.7-a-bis con valores intermedios {50,75,100,150,200}.\n"
                     "Documentar como restricción explícita del método.")

    elif std_fails and std_diverging and not falla_n_alto:
        scenario = "S10.3"
        veredicto = "INESTABILIDAD ESCALANTE"
        next_step = ("std supera el umbral y crece con N aunque la calibración promedio se mantiene.\n"
                     "Requiere análisis específico antes de v7.7-b.\n"
                     "Posible v7.7-a-aux con valores intermedios de N.")

    elif all_cumplen and std_ok_todos and errcal_increasing:
        scenario = "INTERMEDIO"
        veredicto = "ROBUSTEZ OPERATIVA CON DERIVA AL ALZA"
        next_step = ("Los criterios cumplen en todos los N, pero ErrCal α=0.50 sube con N.\n"
                     "No declarar S10.1 fuerte sin revisar C10.e.\n"
                     "Posible repetir con valores intermedios si la deriva es relevante.")

    else:
        scenario = "INTERMEDIO"
        veredicto = "PATRÓN NO CLASIFICABLE"
        next_step = ("Patrón mixto. Análisis cualitativo de C10.e requerido.")

    return scenario, veredicto, next_step


# ==============================================================================
# RUNNER GENÉRICO POR STAGE
# ==============================================================================

def run_stage_for_n(n_ensemble, output_filename):
    """
    Ejecuta evaluación completa para un valor de N (principal + control)
    y guarda resultados en .pkl para uso posterior por stage5.
    """
    print("="*80)
    print(f"MTPS-C v7.7-a — STAGE para N_ensemble = {n_ensemble}")
    print("="*80)
    print(f"\nMÉTODO aux2b CONGELADO. Solo varía N_ensemble.")
    print(f"Dataset mixto: 60 configs (20 ID + 20 OOD cercano + 20 OOD lejano)")
    print(f"N_SPLITS={N_SPLITS}. M={M_GROUND_TRUTHS}. k={K_NEIGHBORS}\n")

    t_total = time.time()

    dataset = generate_mixed_dataset()
    seeds_ens = [50000 + i for i in range(len(dataset))]
    seeds_gt_base = [60000 + i * M_GROUND_TRUTHS for i in range(len(dataset))]

    print(f"Dataset mixto generado: {len(dataset)} configs")
    counts = {}
    for c in dataset:
        counts[c['subset']] = counts.get(c['subset'], 0) + 1
    for s, n in counts.items():
        print(f"  {s}: {n} configs")

    # PRINCIPAL
    print(f"\n{'='*80}\nEJECUTANDO PRINCIPAL (aux2b) con N={n_ensemble}\n{'='*80}")
    all_results_principal = []
    t0 = time.time()
    for idx, config in enumerate(dataset):
        result = evaluate_config_v77a(config, seeds_ens[idx], seeds_gt_base[idx],
                                        'v75b_aux2b', n_ensemble)
        all_results_principal.append(result)
        if (idx + 1) % max(1, len(dataset) // 4) == 0:
            el = time.time() - t0
            eta = el / (idx + 1) * (len(dataset) - idx - 1)
            print(f"  ...{idx+1}/{len(dataset)} ({el:.0f}s, ETA {eta:.0f}s)", flush=True)
    print(f"  Completado: {len(all_results_principal)} en {time.time()-t0:.0f}s")
    aggregated_principal = aggregate_results_by_n(all_results_principal,
                                                    f'PRINCIPAL N={n_ensemble}',
                                                    n_ensemble)

    # CONTROL
    print(f"\n{'='*80}\nEJECUTANDO CONTROL (v75a) con N={n_ensemble}\n{'='*80}")
    all_results_control = []
    t0 = time.time()
    for idx, config in enumerate(dataset):
        result = evaluate_config_v77a(config, seeds_ens[idx], seeds_gt_base[idx],
                                        'v75a_control', n_ensemble)
        all_results_control.append(result)
        if (idx + 1) % max(1, len(dataset) // 4) == 0:
            el = time.time() - t0
            eta = el / (idx + 1) * (len(dataset) - idx - 1)
            print(f"  ...{idx+1}/{len(dataset)} ({el:.0f}s, ETA {eta:.0f}s)", flush=True)
    print(f"  Completado: {len(all_results_control)} en {time.time()-t0:.0f}s")
    aggregated_control = aggregate_results_by_n(all_results_control,
                                                  f'CONTROL N={n_ensemble}',
                                                  n_ensemble)

    total_elapsed = time.time() - t_total
    print(f"\n  Tiempo total stage: {total_elapsed/60:.1f} min")

    # Resumen rápido
    if aggregated_principal:
        print(f"\n  PRINCIPAL N={n_ensemble}:")
        n_ok = sum([aggregated_principal['c10a'], aggregated_principal['c10b'],
                     aggregated_principal['c10c'], aggregated_principal['c10d']])
        print(f"    Criterios cumplidos: {n_ok}/4")
        print(f"    ErrCal α=0.50: {100*aggregated_principal['errcal_global'][P_MAIN]:.2f}pp")
        print(f"    Niveles ≤10pp: {aggregated_principal['n_levels_ok']}/5")
        print(f"    Cov(0.95): {100*aggregated_principal['cov_global'][P_HIGH]:.1f}%")
        print(f"    std α=0.50: {100*aggregated_principal['split_std_errcal_50']:.2f}pp")

    if aggregated_control:
        print(f"\n  CONTROL N={n_ensemble}:")
        print(f"    Niveles ≤10pp: {aggregated_control['n_levels_ok']}/5")
        print(f"    ErrCal α=0.50: {100*aggregated_control['errcal_global'][P_MAIN]:.2f}pp")

    # Guardar .pkl
    payload = {
        'n_ensemble': n_ensemble,
        'principal': aggregated_principal,
        'control': aggregated_control,
        'elapsed_seconds': total_elapsed,
    }
    with open(output_filename, 'wb') as f:
        pickle.dump(payload, f)
    print(f"\n  Resultado guardado: {output_filename}")
    print(f"\n{'='*80}\nStage N={n_ensemble} completado.\n{'='*80}")
    print(f"\n  IMPORTANTE: descargar {output_filename} antes de cerrar Colab.")

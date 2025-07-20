import numpy as np
from scipy.integrate import solve_ivp
import json
import datetime
import matplotlib.pyplot as plt

# Constants for physical and simulation parameters
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c = 2.99792458e8  # Speed of light (m/s)
hbar = 1.0545718e-34  # Reduced Planck constant (J s)
m_n = 1.67e-27  # Neutron mass (kg)
l_p = np.sqrt(hbar * G / c**3)  # Planck length (m)
Lambda = 1.1e-52  # Cosmological constant (m^-2)
T_c = l_p / c  # Planck time (s)
kappa_CTC = 0.813  # Closed timelike curve (CTC) coupling
kappa_J4 = 0.813  # J4 coupling for phase dynamics
kappa_worm = 0.01  # Wormhole coupling
kappa_ent = 1.0  # Entanglement coupling (optimized)
kappa_J6 = 0.01  # J6 coupling
kappa_J6_eff = 1e-33  # Effective J6 coupling
kappa_ZPE = 0.1  # Zero-point energy coupling
kappa_grav = 1e-2  # Gravitational coupling
kappa_grav2 = 0.05  # Gravitational-entanglement coupling
eta_CTC = 0.05  # CTC feedback strength
kappa_g = 0.1  # Gravitational gate coupling
beta_ZPE = 1e-3  # ZPE phase factor
d_c = 1e-9  # Characteristic length (m)
lambda_v = 0.33333333326  # Vertex scaling for Tetbit
beta = 1e-3  # Scalar field coupling
theta = 1.79  # Wave number parameter
k = 1 / theta  # Wave number
omega = 2 * np.pi / (100 * 1e-12)  # Angular frequency (rad/s)

# JOSEPHSON JUNCTION PARAMETERS
e_charge = 1.602e-19  # Electron charge (C)
I_c = 1e-6  # Critical current (A) - adjustable based on simulation needs
E_J = (hbar * I_c) / (2 * e_charge)  # Josephson energy (J)
C_j = 1e-15  # Junction capacitance (F)
R_n = 100  # Normal resistance (Ω)

# 6D grid setup
nx, ny, nz, nt, nw1, nw2 = 5, 5, 5, 5, 3, 3  # Dimensions: x, y, z, t, w1, w2
N = nx * ny * nz * nt * nw1 * nw2  # Total grid points: 5625
dx = l_p * 1e5  # Spatial step (m)
dt = 1e-12  # Time step (s)
dw = l_p * 1e3  # Extra dimension step (m)
grid_shape = (nx, ny, nz, nt, nw1, nw2)

V0 = 10 * (hbar**2 / (2 * m_n * dx**2))  # Barrier height for JJ
sigma_barrier = dx  # Barrier width
x0 = 0.0  # Barrier center

config = {
    'phase_shift': np.exp(1j * np.pi / 3),  # Phase shift for Tetbit and MetatronCircle
    'tetbit_scale': 1.0,  # Scaling for Tetbit gates
    'scaling_factor': 1e-3,  # Global scaling factor
    'vertex_lambda': lambda_v  # Vertex scaling for spacetime encoding
}

# Initialize Gaussian wave function shifted to left side (for JJ)
x = np.linspace(-5*dx, 5*dx, nx)
y = np.linspace(-5*dx, 5*dx, ny)
z = np.linspace(-5*dx, 5*dx, nz)
t = np.linspace(0, 5*dt, nt)
w1 = np.linspace(-5*dw, 5*dw, nw1)
w2 = np.linspace(-5*dw, 5*dw, nw2)
X, Y, Z, T, W1, W2 = np.meshgrid(x, y, z, t, w1, w2, indexing='ij')

# Shift initial wavefunction to left side (x = -2.5*dx)
x_shifted = X + 2.5*dx
r_6d = np.sqrt(x_shifted**2 + Y**2 + Z**2 + 0.1*(T**2 + W1**2 + W2**2))
sigma = np.sqrt(hbar / 2)
alpha_values = np.array([0j])  # Assuming placeholder value
psi = np.exp(-r_6d**2 / (2 * sigma**2)) * np.exp(1j * np.sqrt(2) * np.imag(alpha_values[0]) * r_6d / hbar)
psi /= np.linalg.norm(psi)  # Normalize wave function
psi_initial = psi.copy()
psi_past = psi.copy()
phi_N = np.random.uniform(-0.1, 0.1, (nx, ny, nz))
bit_flips = 0
coords = np.array(np.unravel_index(np.arange(N), grid_shape)).T

# Barrier potential for Josephson junction
barrier = V0 * np.exp(-X**2 / (2 * sigma_barrier**2))

class DataLogger:
    """Logs simulation data to JSON file for analysis."""
    def __init__(self, log_file="simulation_log.json"):
        self.log_file = log_file
        self.data = []

    def log(self, data):
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {"timestamp": timestamp, **data}
        self.data.append(log_entry)
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.data, f, indent=4)
        except Exception as e:
            print(f"Logging failed: {e}")

class Tetbit:
    """4-state quantum system (ququart) with tetrahedral symmetry for spacetime encoding."""
    def __init__(self, config, position=None):
        self.config = config
        self.phase_shift = config["phase_shift"]
        self.state = np.zeros(4, dtype=np.complex128)
        self.state[0] = 1.0  # Initial state
        self.prev_state = self.state.copy()  # For bit flip detection
        self.y_gate = self._tetrahedral_y()
        self.h_gate = self._tetrahedral_hadamard()
        if position is not None:
            self.encode_spacetime_position(position)

    def _tetrahedral_y(self):
        """Tetrahedral Y-gate with phase shift for 4-state rotations."""
        return np.array([[0, 0, 0, -self.phase_shift * 1j], [self.phase_shift * 1j, 0, 0, 0],
                        [0, self.phase_shift * 1j, 0, 0], [0, 0, self.phase_shift * 1j, 0]], dtype=np.complex128)

    def _tetrahedral_hadamard(self):
        """Tetrahedral Hadamard gate using golden ratio for superposition."""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        scale = self.config['tetbit_scale'] * self.config['scaling_factor']
        h = np.array([[1, 1, 1, 1], [1, phi, -1/phi, -1], [1, -1/phi, phi, -1], [1, -1, -1, 1]], dtype=np.complex128) * scale
        norm = np.linalg.norm(h, axis=0)
        return h / norm[np.newaxis, :]

    def apply_gate(self, gate):
        """Apply quantum gate and normalize state."""
        self.state = gate @ self.state
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

    def measure(self):
        """Measure state, detect bit flips, and collapse to basis state."""
        probs = np.abs(self.state)**2
        probs_sum = np.sum(probs)
        if probs_sum > 0:
            probs /= probs_sum
        else:
            probs = np.ones(4) / 4
        outcome = np.random.choice(4, p=probs)
        bit_flip = not np.allclose(self.state, self.prev_state, atol=1e-8)
        self.prev_state = self.state.copy()
        self.state = np.zeros(4, dtype=np.complex128)
        self.state[outcome] = 1.0
        return outcome, bit_flip

    def encode_spacetime_position(self, position, t_path=0):
        """Encode 3D position into tetrahedral quantum state."""
        vertices = np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]]) * self.config['vertex_lambda'] * self.config['scaling_factor']
        distances = np.linalg.norm(vertices - position[:3], axis=1)
        weights = np.exp(-distances**2 / (2 * self.config['tetbit_scale']**2))
        phase_adj = np.exp(1j * np.pi / 3 * t_path)
        total_weight = np.sum(weights)
        if total_weight > 0:
            self.state = (weights.astype(np.complex128) / np.sqrt(total_weight)) * phase_adj
        return self.state

class MetatronCircle:
    """Geometric configuration for global phase shifts in 6D simulation."""
    def __init__(self, config, center=np.array([0.0, 0.0, 0.0]), radius=1.0):
        self.config = config
        self.center = center
        self.radius = radius
        self.points = self._generate_points(0)
        self.phase_shift = config["phase_shift"]
        self.state = np.ones(len(self.points), dtype=np.complex128) / np.sqrt(len(self.points))

    def _generate_points(self, t_idx, singularity_alpha=0.01):
        """Generate 13 points on a 2D circle in 3D space."""
        theta = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        radius = self.radius * np.exp(-singularity_alpha * t_idx)  # Drive to singularity
        x = self.center[0] + radius * np.cos(theta)
        y = self.center[1] + radius * np.sin(theta)
        z = np.zeros_like(theta) + self.center[2]
        return np.column_stack((x, y, z))

    def apply_phase_shift(self):
        """Apply global phase shift to state."""
        self.state *= self.phase_shift
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

def scalar_field(r_6d, t):
    """Scalar field for wave function modulation."""
    return -r_6d**2 * np.cos(k * r_6d - omega * t) + 2 * r_6d * np.sin(k * r_6d - omega * t) + 2 * np.cos(k * r_6d - omega * t)

def nugget_field_deriv(t, phi_N_flat):
    """Evolve scalar field phi_N with Laplacian and nonlinear terms."""
    phi_N = phi_N_flat.reshape((nx, ny, nz))
    dphi_N_dt = np.zeros_like(phi_N)
    d2phi_N_dt2 = np.zeros_like(phi_N)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                laplacian = 0
                for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                    ni, nj, nk = (i+di)%nx, (j+dj)%ny, (k+dk)%nz
                    laplacian += (phi_N[ni, nj, nk] - phi_N[i, j, k]) / dx**2
                laplacian -= 6 * phi_N[i, j, k] / dx**2
                nonlinear = phi_N[i, j, k] * (phi_N[i, j, k]**2 - 1) * (1 + 0.1 * np.sin(2 * np.pi * t))
                d2phi_N_dt2[i, j, k] = laplacian - 0.1**2 * phi_N[i, j, k] + 0.5 * phi_N[i, j, k] - nonlinear - (1/c**2) * dphi_N_dt[i, j, k]
    return np.concatenate([dphi_N_dt.flatten(), d2phi_N_dt2.flatten()])

def compute_morley_adjustment(coords):
    """Compute geometric adjustment for fractal dimension."""
    s = 4 * dx
    v1, v2, v3 = np.array([0, 0, 0]), np.array([s, 0, 0]), np.array([s/2, (np.sqrt(3)/2)*s, 0])
    morley_centroid = (v1 + v2 + v3) / 3
    distances = np.linalg.norm(coords[:, :3] - morley_centroid, axis=1)
    morley_h = (np.sqrt(3)/2) * (s / np.sqrt(3))
    adjustment = 0.05 * (morley_h - distances.mean())**2
    return adjustment

def fractal_dimension_max(phi_N, r, t, coords):
    """Compute fractal dimension for lattice dynamics."""
    grad_phi_N = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                grad_x = (phi_N[(i+1)%nx, j, k] - phi_N[(i-1)%nx, j, k]) / (2 * dx)
                grad_y = (phi_N[i, (j+1)%ny, k] - phi_N[i, (j-1)%ny, k]) / (2 * dx)
                grad_z = (phi_N[i, j, (k+1)%nz] - phi_N[i, j, (k-1)%nz]) / (2 * dx)
                grad_phi_N[i, j, k] = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    d_f = 1.7 + 0.3 * np.tanh(np.abs(grad_phi_N)**2 / 0.1) + 0.05 * np.log(1 + r/l_p) * np.cos(2 * np.pi * t / T_c)
    multi_scale = sum(alpha * (r / (l_p * 10**(k-1)))**(d_f-3) * np.sin(2 * np.pi * t / (T_c * 10**(k-1)))
                      for k, alpha in enumerate([0.02, 0.01, 0.005], 1))
    d_f += multi_scale
    morley_adjustment = compute_morley_adjustment(coords)
    d_f += morley_adjustment
    return d_f

def zpe_density_max(r, phi, sigma, g_00, psi, psi_past, dx, t, d_f):
    """Compute zero-point energy density with fractal and CTC effects."""
    rho_0 = -0.5 * hbar * c / d_c**4
    R_r = 1 + r/l_p + 0.2 * (r/l_p)**2
    A_phi = np.cos(phi)**2 + np.sin(phi)**2 + 0.1 * np.sin(2*phi)
    E_sigma = 1 + 0.15 * sigma / np.log2(N**3) * 1.618
    C_g = 1 + 0.05 * g_00 / (-c**2)
    F_r = -1.5e-17
    grad_psi = np.sum(np.abs(np.gradient(psi, dx))**2)
    r_squeeze = 0.1
    N_phi_N = 1 + 0.2 * np.exp(r_squeeze) * np.mean(np.abs(np.gradient(phi_N, dx))**2) * (r/l_p)**(d_f-3) * (1 + 0.1 * np.sin(2 * np.pi * np.log(r/l_p + 1e-10)))
    rho_zpe = rho_0 * R_r * A_phi * E_sigma * C_g * (1 + 0.1 * abs(F_r) / abs(-3.2e-17)) * N_phi_N
    ctc_feedback = eta_CTC * np.abs(psi_past)**2 / (np.abs(psi)**2 + 1e-10) * np.exp(-np.abs(np.angle(psi) - np.angle(psi_past)) / T_c)
    return rho_zpe * (1 + kappa_ZPE * grad_psi / (hbar**2 / d_c**2) + ctc_feedback)

def compute_ricci_scalar(r_6d, g_00):
    """Compute Ricci scalar for spacetime curvature."""
    R = -2 * (1 / r_6d**2) * (1 + g_00 / c**2)
    return R

def godel_metric_max(r_6d, psi, psi_past, t):
    """Compute Godel-like metric with fractal and CTC effects."""
    d_f = fractal_dimension_max(phi_N, r_6d, t, coords)
    g_00 = -1 + 1e-5 * np.abs(psi)**2 * np.sin(k * r_6d - omega * t)
    ctc_term = 1e-4 * d_f * np.abs(psi_past)**2 / (np.abs(psi)**2 + 1e-10) * np.exp(1j * T_c * np.tanh(np.angle(psi) - np.angle(psi_past)))
    ricci = compute_ricci_scalar(r_6d, g_00 + ctc_term)
    return np.diag([g_00 + ctc_term, 1 + ctc_term, 1 + ctc_term, np.sinh(2 * r_6d)**2 + ctc_term]), ricci

def gravitational_qubit(psi, psi_past, r_6d, t):
    """Apply gravitational modulation to wave function."""
    d_f = fractal_dimension_max(phi_N, r_6d, t, coords)
    g_00 = -1 + 1e-5 * np.abs(psi)**2 * np.sin(k * r_6d - omega * t) + 1e-4 * d_f * np.abs(psi_past)**2 / (np.abs(psi)**2 + 1e-10) * np.exp(1j * T_c * np.tanh(np.angle(psi) - np.angle(psi_past)))
    return np.sqrt(np.abs(g_00)) * psi

def zpe_amplified_gate(psi, psi_past, r_6d, phi, sigma, t):
    """Apply zero-point energy amplified gate with fractal effects."""
    d_f = fractal_dimension_max(phi_N, r_6d, t, coords)
    zpe = zpe_density_max(r_6d, phi, sigma, -1, psi, psi_past, dx, t, d_f)
    dV = dx**3 * dt * dw**2
    U_g, ricci = godel_metric_max(r_6d, psi, psi_past, t)
    U_zpe = np.exp(1j * beta_ZPE * np.sum(zpe) * dV)
    return U_zpe * np.eye(N, dtype=np.complex128), ricci

def hamiltonian_grav_ent_max(psi, d_f):
    """Compute gravitational-entanglement Hamiltonian term."""
    r_6d = np.sqrt(np.sum([(coords[:,i] - 2)**2 * [1, 1, 1, 0.1, 0.1, 0.1][i] for i in range(6)], axis=0))
    psi_abs_sq = np.abs(psi.ravel())**2
    H_grav_ent = np.zeros_like(psi)
    cutoff = 5.0  # Distance cutoff for interactions
    for idx in range(N):
        pos = np.unravel_index(idx, grid_shape)
        r_diff = np.sqrt(np.sum([(coords[idx] - coords)**2 * [1, 1, 1, 0.1, 0.1, 0.1]], axis=1))
        mask = (r_diff < cutoff) & (r_diff > 1e-10)
        sum_term = np.sum(psi_abs_sq[mask] * (1/r_diff[mask]**4 + d_f * np.exp(-r_diff[mask]/l_p)) / (r_diff[mask]**4 + 1e-10))
        H_grav_ent[pos] = (kappa_grav + kappa_grav2 * d_f) * G * m_n * sum_term * psi[pos]
    return H_grav_ent

def hamiltonian(psi, psi_past, phi_N, t, paths):
    """Compute 6D Hilbert Hamiltonian with fractal and exotic terms."""
    H_kin = np.zeros_like(psi, dtype=np.complex128)
    H_pot = np.zeros_like(psi, dtype=np.complex128)
    H_worm = np.zeros_like(psi, dtype=np.complex128)
    H_ent = np.zeros_like(psi, dtype=np.complex128)
    H_CTC = np.zeros_like(psi, dtype=np.complex128)
    H_J4 = np.zeros_like(psi, dtype=np.complex128)
    r_6d = np.sqrt(np.sum([(coords[:,i] - 2)**2 * [1, 1, 1, 0.1, 0.1, 0.1][i] for i in range(6)], axis=0))
    phi = np.arctan2(coords[:,1], coords[:,0])
    sigma = 0.5 * np.log2(N**3)
    d_f = fractal_dimension_max(phi_N, r_6d, t, coords)
    r_squeeze = 0.1
    omega_h = 1e12
    
    # JOSEPHSON BARRIER: Add potential barrier at x=0
    H_pot += barrier
    
    delta = [dx, dx, dx, dt, dw, dw]
    for idx in range(N):
        i, j, k, l, m, n = np.unravel_index(idx, grid_shape)
        for d in range(6):
            pos_plus = list((i, j, k, l, m, n))
            pos_plus[d] = (pos_plus[d] + 1) % grid_shape[d]
            pos_minus = list((i, j, k, l, m, n))
            pos_minus[d] = (pos_minus[d] - 1) % grid_shape[d]
            H_kin.flat[idx] += -(hbar**2 / (2 * m_n)) * (psi[tuple(pos_plus)] + psi[tuple(pos_minus)] - 2*psi.flat[idx]) / delta[d]**2
        x = [coords[idx, d] * delta[d] for d in range(6)]
        p = [-1j * hbar * (psi[tuple(pos_plus)] - psi[tuple(pos_minus)]) / (2 * delta[d]) for d in range(6)]
        V_grav = -G * m_n / (r_6d[idx]**4 * Lambda**2) * (1 + 1e-3 * (-np.sum(np.abs(psi.ravel())**2 * np.log(np.abs(psi.ravel())**2 + 1e-10))))
        H_pot.flat[idx] += V_grav * (1 + 2 * np.sin(t)) + sum(0.5 * hbar * omega_h * (x[d]**2 + np.abs(p[d])**2) + 0.5 * hbar * r_squeeze * (x[d] * p[d] + p[d] * x[d]) for d in range(6)) * psi.flat[idx]
        H_worm.flat[idx] = kappa_worm * np.exp(1j * 2 * t) * np.exp(-r_6d[idx]**2 / (2 * 1.0**2)) * psi.flat[idx]
        ent_sum = 0
        for d in range(6):
            pos_neighbor = list((i, j, k, l, m, n))
            pos_neighbor[d] = (pos_neighbor[d] + 1) % grid_shape[d]
            ent_sum += kappa_ent * (1 + np.sin(t)) * (psi[tuple(pos_neighbor)] - psi.flat[idx]) * np.conj(psi[tuple(pos_neighbor)] - psi.flat[idx])
        H_ent.flat[idx] = ent_sum
        H_CTC.flat[idx] = kappa_CTC * np.exp(1j * T_c * np.tanh(np.angle(psi.flat[idx]) - np.angle(psi_past.flat[idx]))) * abs(psi.flat[idx])
        H_J4.flat[idx] = kappa_J4 * np.sin(np.angle(psi.flat[idx])) * psi.flat[idx]
    for path_idx in range(len(paths)):
        path_tuple = tuple(paths[path_idx])
        H_CTC[path_tuple] += kappa_CTC * np.exp(1j * T_c * np.tanh(np.angle(psi[path_tuple]) - np.angle(psi_past[path_tuple]))) * abs(psi[path_tuple]) * np.sin(t + path_idx)
    return H_kin + H_pot + H_worm + H_ent + H_CTC + H_J4 + hamiltonian_grav_ent_max(psi, d_f)

def compute_cv_entanglement(psi, coords, t):
    """Compute logarithmic negativity and von Neumann entropy for entanglement."""
    n_modes = N
    cov = np.zeros((12, 12))
    for i in range(n_modes):
        idx = np.unravel_index(i, grid_shape)
        x = [coords[i, d] * [dx, dx, dx, dt, dw, dw][d] for d in range(6)]
        p = []
        for d in range(6):
            idx_plus = list(idx)
            idx_plus[d] = (idx_plus[d] + 1) % grid_shape[d]
            idx_minus = list(idx)
            idx_minus[d] = (idx_minus[d] - 1) % grid_shape[d]
            p_d = -1j * hbar * (psi[tuple(idx_plus)] - psi[tuple(idx_minus)]) / (2 * [dx, dx, dx, dt, dw, dw][d])
            p.append(p_d)
        for d in range(6):
            cov[2*d, 2*d] = np.abs(x[d])**2
            cov[2*d+1, 2*d+1] = np.abs(p[d])**2
            if i < n_modes - 1:
                idx2 = np.unravel_index(i + 1, grid_shape)
                x2 = [coords[i + 1, d] * [dx, dx, dx, dt, dw, dw][d] for d in range(6)]
                p2 = []
                for dd in range(6):
                    idx2_plus = list(idx2)
                    idx2_plus[dd] = (idx2_plus[dd] + 1) % grid_shape[dd]
                    idx2_minus = list(idx2)
                    idx2_minus[dd] = (idx2_minus[dd] - 1) % grid_shape[dd]
                    p2_d = -1j * hbar * (psi[tuple(idx2_plus)] - psi[tuple(idx2_minus)]) / (2 * [dx, dx, dx, dt, dw, dw][dd])
                    p2.append(p2_d)
                phase = np.exp(1j * np.pi * t / T_c)  # Temporal vector lattice
                cov[2*d, 2*d+6] = np.real(np.conj(x[d]) * x2[d] * phase)
                cov[2*d+1, 2*d+7] = np.real(np.conj(p[d]) * p2[d] * phase)
    det = np.linalg.det(cov)
    log_neg = -np.log(max(det, 1e-10)) if det > 0 else 0
    probs = np.abs(psi.ravel())**2
    probs /= np.sum(probs) + 1e-10
    von_neumann = -np.sum(probs * np.log2(probs + 1e-10))
    return log_neg, von_neumann

def compute_bell_inequality(psi):
    """Compute simplified CHSH-like Bell inequality for entanglement verification."""
    probs = np.abs(psi.ravel())**2
    probs /= np.sum(probs) + 1e-10
    idx1, idx2 = np.random.choice(N, 2, replace=False)
    psi1, psi2 = psi.ravel()[idx1], psi.ravel()[idx2]
    E_ab = np.real(np.conj(psi1) * psi2)
    E_aB = np.imag(np.conj(psi1) * psi2)
    E_Ab = np.real(np.conj(psi1) * psi2 * np.exp(1j * np.pi / 4))
    E_AB = np.imag(np.conj(psi1) * psi2 * np.exp(1j * np.pi / 4))
    chsh = abs(E_ab + E_aB + E_Ab - E_AB)
    return chsh > 2

def compute_fidelity(psi, psi_initial):
    """Compute fidelity to measure state preservation."""
    return np.abs(np.sum(np.conj(psi_initial.ravel()) * psi.ravel()))**2 / (np.linalg.norm(psi_initial)**2 * np.linalg.norm(psi)**2)

def plot_probability_density(psi, t_idx, coords):
    """Visualize probability density in 3D projection."""
    probs = np.abs(psi.ravel())**2
    probs /= np.sum(probs) + 1e-10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=probs, cmap='viridis')
    plt.colorbar(sc, label='Probability Density')
    ax.set_title(f'Probability Density at Step {t_idx}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(f'probability_density_step_{t_idx}.png')
    plt.close()

def mobius_spiral_trajectory_6d(t, r, n):
    theta = t * np.pi
    phi1 = n * theta / 2
    phi2 = theta / 3
    phi3 = theta / 4
    phi4 = theta / 5
    x = r * np.cos(theta) * np.sin(phi1) * np.cos(phi2) * np.sin(phi3) * np.cos(phi4)
    y = r * np.sin(theta) * np.sin(phi1) * np.cos(phi2) * np.sin(phi3) * np.cos(phi4)
    z = r * np.cos(phi1) * np.cos(phi2) * np.sin(phi3) * np.cos(phi4)
    w1 = r * np.sin(phi2) * np.sin(phi3) * np.cos(phi4)
    w2 = r * np.cos(phi3) * np.cos(phi4)
    time = r * np.sin(phi4)
    return np.stack((x, y, z, time, w1, w2), axis=-1)

def compute_ctc_wormhole_paths(n_points=50):
    t = np.linspace(0, 2 * np.pi, n_points)
    paths = mobius_spiral_trajectory_6d(t, 3*dx, 2)
    # Map to grid
    grid_min = np.array([-5*dx, -5*dx, -5*dx, 0, -5*dw, -5*dw])
    step = np.array([dx, dx, dx, dt, dw, dw])
    indices = np.floor((paths - grid_min) / step).astype(int) % np.array(grid_shape)
    return indices, t

# Simulation parameters
logger = DataLogger()
n_iterations = 50
runs = 1
entanglement_history = []
von_neumann_history = []
fidelity_history = []
d_f_history = []
zpe_history = []
bit_flip_history = []
ricci_history = []
bell_violation_count = 0

# JOSEPHSON JUNCTION TRACKING VARIABLES
phi_angle_history = []  # Phase difference history
N_right_history = []    # Probability on right side
dN_right_dt = []        # Derivative of right probability
I_current = []          # Josephson current
V_junction = []         # Voltage across junction

# Add CTC paths
paths, t_path = compute_ctc_wormhole_paths()

# Main simulation loop
for run in range(runs):
    # Initialize wavefunction on left side
    x_shifted = X + 2.5*dx
    r_6d = np.sqrt(x_shifted**2 + Y**2 + Z**2 + 0.1*(T**2 + W1**2 + W2**2))
    psi = np.exp(-r_6d**2 / (2 * sigma**2)) * np.exp(1j * np.sqrt(2) * np.imag(alpha_values[0]) * r_6d / hbar)
    psi /= np.linalg.norm(psi)
    psi_initial = psi.copy()
    psi_past = psi.copy()
    phi_N = np.random.uniform(-0.1, 0.1, (nx, ny, nz))
    bit_flips = 0
    
    # Initialize Josephson tracking
    N_right_prev = None
    
    for t_idx in range(n_iterations):
        t_val = t_idx * dt
        phi_N_flat = phi_N.flatten()
        sol = solve_ivp(nugget_field_deriv, [t_val, t_val + dt], np.concatenate([phi_N_flat, np.zeros_like(phi_N_flat)]), method='RK45')
        phi_N = sol.y[:phi_N.size, -1].reshape((nx, ny, nz))
        
        r_6d = np.sqrt(np.sum([(coords[:,i] - 2)**2 * [1, 1, 1, 0.1, 0.1, 0.1][i] for i in range(6)], axis=0))
        phi = np.arctan2(coords[:,1], coords[:,0])
        sigma = 0.5 * np.log2(N**3)
        
        d_f = fractal_dimension_max(phi_N, r_6d, t_val, coords)
        psi_g = gravitational_qubit(psi, psi_past, r_6d, t_val)
        U_g, ricci = zpe_amplified_gate(psi, psi_past, r_6d, phi, sigma, t_val)
        psi = (U_g @ psi_g.ravel()).reshape(grid_shape)
        psi *= np.exp(1j * beta * scalar_field(r_6d, t_val))
        H = hamiltonian(psi, psi_past, phi_N, t_val, paths)
        psi = psi - 1j * dt / hbar * H
        psi /= np.linalg.norm(psi)
        
        tet = Tetbit(config, position=np.array([0, 0, 0]))
        tet.apply_gate(tet.y_gate)
        tet.encode_spacetime_position(np.array([0,0,0]), t_path[t_idx % len(t_path)])
        for idx in range(N):
            if np.random.rand() < 0.1:
                outcome, flip = tet.measure()
                if flip:
                    bit_flips += 1
                psi.ravel()[idx] *= tet.state[outcome]
        
        metatron = MetatronCircle(config)
        metatron.points = metatron._generate_points(t_idx)
        metatron.apply_phase_shift()
        path_index = int(13 * (t_path[t_idx % len(t_path)] / (2*np.pi)) % 13)
        psi *= metatron.state[path_index]
        
        zpe = zpe_density_max(r_6d, phi, sigma, -1, psi, psi_past, dx, t_val, d_f)
        log_neg, von_neumann = compute_cv_entanglement(psi, coords, t_val)
        fidelity = compute_fidelity(psi, psi_initial)
        bell_violation = compute_bell_inequality(psi)
        if bell_violation:
            bell_violation_count += 1
        
        # JOSEPHSON JUNCTION CALCULATIONS
        mask_A = (X < x0)  # Left superconductor
        mask_B = (X >= x0)  # Right superconductor
        
        # Total amplitude in each region
        psi_A = np.sum(psi[mask_A])
        psi_B = np.sum(psi[mask_B])
        
        # Phase difference
        phase_A = np.angle(psi_A)
        phase_B = np.angle(psi_B)
        phi_angle = phase_B - phase_A
        phi_angle_history.append(phi_angle)
        
        # Probability on right side
        N_right_current = np.sum(np.abs(psi[mask_B])**2)
        N_right_history.append(N_right_current)
        
        # Current calculation (dN_right/dt)
        if N_right_prev is not None:
            current = (N_right_current - N_right_prev) / dt
            dN_right_dt.append(current)
            I_current.append(current * (2 * e_charge))  # Convert to electrical current
        else:
            dN_right_dt.append(0.0)
            I_current.append(0.0)
        N_right_prev = N_right_current
        
        # Voltage calculation (using second Josephson equation)
        if t_idx > 0:
            dphi_dt = (phi_angle - phi_angle_history[-2]) / dt
            V_junction.append((hbar / (2 * e_charge)) * dphi_dt)
        else:
            V_junction.append(0.0)
        
        # Visualize probability density every 10 steps
        if t_idx % 10 == 0:
            plot_probability_density(psi, t_idx, coords)
        
        logger.log({
            "run": run,
            "step": t_idx,
            "phi_angle": float(phi_angle),
            "N_right": float(N_right_current),
            "dN_right_dt": float(dN_right_dt[-1]),
            "I_josephson": float(I_current[-1]),
            "V_junction": float(V_junction[-1]),
            "log_negativity": float(log_neg),
            "von_neumann_entropy": float(von_neumann),
            "fidelity": float(fidelity),
            "avg_d_f": float(np.mean(d_f)),
            "avg_zpe": float(np.mean(zpe)),
            "bit_flips": bit_flips,
            "avg_ricci": float(np.mean(ricci)),
            "bell_violation": bell_violation
        })
        
        entanglement_history.append(log_neg)
        von_neumann_history.append(von_neumann)
        fidelity_history.append(fidelity)
        d_f_history.append(np.mean(d_f))
        zpe_history.append(np.mean(zpe))
        bit_flip_history.append(bit_flips)
        ricci_history.append(np.mean(ricci))
        
        psi_past = psi.copy()

# Summarize results
avg_log_neg = np.mean(entanglement_history)
std_log_neg = np.std(entanglement_history)
avg_von_neumann = np.mean(von_neumann_history)
print(f"Optimized Simulation Results (1 run, 50 iterations, N={N}):")
print(f"Avg Log Negativity: {avg_log_neg:.3f} ± {std_log_neg:.3f}")
print(f"Avg von Neumann Entropy: {avg_von_neumann:.3f}")
print(f"Final Fidelity: {fidelity_history[-1]:.3f}")
print(f"Avg d_f: {np.mean(d_f_history):.3f}")
print(f"Avg ZPE: {np.mean(zpe_history):.3e}")
print(f"Bit Flips: {bit_flip_history[-1]}")
print(f"Avg Ricci: {np.mean(ricci_history):.3e}")
print(f"Bell Violations: {bell_violation_count}")

# Plot Josephson junction dynamics
plt.figure(figsize=(12, 8))

# Current vs phase difference
plt.subplot(221)
plt.plot(phi_angle_history, I_current, 'b.')
plt.xlabel('Phase Difference φ (rad)')
plt.ylabel('Josephson Current (A)')
plt.title('I vs φ')

# Current vs time
plt.subplot(222)
plt.plot(np.arange(n_iterations)*dt, I_current)
plt.xlabel('Time (s)')
plt.ylabel('Josephson Current (A)')
plt.title('Current Evolution')

# Voltage vs time
plt.subplot(223)
plt.plot(np.arange(n_iterations)*dt, V_junction)
plt.xlabel('Time (s)')
plt.ylabel('Voltage (V)')
plt.title('Voltage Evolution')

# Phase difference vs time
plt.subplot(224)
plt.plot(np.arange(n_iterations)*dt, phi_angle_history)
plt.xlabel('Time (s)')
plt.ylabel('Phase Difference φ (rad)')
plt.title('Phase Evolution')

plt.tight_layout()
plt.savefig('josephson_dynamics.png')
plt.close()

# Plot standard results
plt.plot(np.arange(n_iterations), entanglement_history, label='Log Negativity')
plt.xlabel('Iteration')
plt.ylabel('Log Negativity')
plt.title('Entanglement Entropy Evolution')
plt.legend()
plt.savefig('entanglement_entropy_scaled.png')
plt.close()

plt.plot(np.arange(n_iterations), von_neumann_history, label='von Neumann Entropy')
plt.xlabel('Iteration')
plt.ylabel('von Neumann Entropy')
plt.title('von Neumann Entropy Evolution')
plt.legend()
plt.savefig('von_neumann_entropy_scaled.png')
plt.close()
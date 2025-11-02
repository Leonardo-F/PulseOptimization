import numpy as np
import qutip as qt
from scipy.interpolate import interp1d
from typing import Dict, Tuple, List
import json
import time
from multiprocessing import Pool

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="qutip")

# 定义工作函数, 用于并行计算
def worker(args):
    self, pulses, phi, psi_i, n_shots, seed = args
    return self.simulate_evolution_ensemble(pulses, phi, psi_i, n_shots, seed)

class TransmonPulseGrader:
    """
    Grader for evaluating √X gate pulses in a transmon system

    This grader simulates the pulse evolution using QuTiP with open system dynamics
    and evaluates the pulse quality based on multiple criteria including gate fidelity,
    leakage suppression, and pulse quality constraints.
    """
    
    def __init__(self, 
                 n_levels: int = 4,
                 n_steps: int = 30,  # Number of time steps
                 alpha: float = -2 * np.pi * 0.2e9,  # Anharmonicity (rad/s)
                 omega_q: float = 2 * np.pi * 5.0e9,  # Qubit frequency (rad/s)
                 omega_d: float = 2 * np.pi * 5.0e9,  # Drive frequency (rad/s)
                 dt: float = 5e-10,  # Time step (s)
                 T1: float = 50e-6,  # Energy relaxation time (s)
                 T_phi: float = 30e-6,  # Pure dephasing time (s)
                 n_bar: float = 0.05,  # Thermal occupation
                 sigma_freq: float = 0.5e6,  # Frequency drift noise std (Hz)
                 n_shots: int = 15,  # Number of shots for averaging
                 h_a: float = 179e6,  # Amplitude threshold (Hz)
                 h_d: float = 22.4e6,  # Derivative threshold (Hz)
                 A_penalty: float = 0.1,  # Penalty scaling factor
                 computing_method: str = 'serial' # 'parallel' or 'serial' 计算损失值时使用串行或并行方法，默认是串行
                 ):  # Penalty scaling factor
        """
        Parameters:
        -----------
        n_levels : int
            Number of transmon energy levels
        n_steps : int
            Expected number of time steps in pulse sequences
        alpha : float
            Anharmonicity in rad/s
        omega_q : float
            Qubit frequency in rad/s
        omega_d : float
            Drive frequency in rad/s
        dt : float
            Time step in seconds
        T1 : float
            Energy relaxation time in seconds
        T_phi : float
            Pure dephasing time in seconds
        n_bar : float
            Thermal occupation number
        sigma_freq : float
            Standard deviation of frequency drift noise in Hz
            This represents shot-to-shot fluctuations with ⟨δu_f⟩=0, ⟨(δu_f)²⟩=σ_f²
        n_shots : int
            Number of shots for ensemble averaging
        h_a : float
            Amplitude threshold in Hz (will be converted to rad/s internally)
        h_d : float
            Derivative threshold in Hz (will be converted to rad/s internally)
        A_penalty : float
            Penalty scaling factor
        """
        
        # 计算损失值时使用的方法，默认是串行运算    
        self.computing_method = computing_method


        self.n_levels = n_levels
        self.n_steps = n_steps  # Store expected number of steps
        self.alpha = alpha
        self.omega_q = omega_q
        self.omega_d = omega_d
        self.delta = omega_d - omega_q
        self.dt = dt
        
        # Noise parameters
        self.T1 = T1
        self.T_phi = T_phi
        self.n_bar = n_bar
        self.sigma_freq = sigma_freq  # Frequency drift noise std
        self.n_shots = n_shots  # Number of shots for averaging
        
        # Penalty parameters - Convert Hz to rad/s for unit consistency
        self.h_a = h_a * 2 * np.pi  # Convert Hz to rad/s
        self.h_d = h_d * 2 * np.pi  # Convert Hz to rad/s
        self.A_penalty = A_penalty
        
        # Store original values in Hz for reporting
        self.h_a_hz = h_a
        self.h_d_hz = h_d
        
        # Create QuTiP operators
        self.a = qt.destroy(n_levels)
        self.a_dag = qt.create(n_levels)
        self.n_op = self.a_dag * self.a
        
        # Base static Hamiltonian (anharmonicity only, drift added per shot)
        self.H_static_base = 0.5 * alpha * (self.a_dag**2) * (self.a**2)
        
        # Collapse operators (same for all shots)
        self.c_ops = self._setup_collapse_operators()
        
        # Target gate: √X = R_X(π/2)
        self.target_gate_2x2 = self._create_target_gate()
        
        # Cardinal states for evaluation
        self.cardinal_states = self._create_cardinal_states()
    
    def _setup_collapse_operators(self) -> List[qt.Qobj]:
        """Setup Lindblad collapse operators"""
        c_ops = []
        
        if self.T1 > 0:
            # Thermal relaxation
            gamma_down = (1 + self.n_bar) / self.T1
            c_ops.append(np.sqrt(gamma_down) * self.a)
            
            # Thermal excitation
            if self.n_bar > 0:
                gamma_up = self.n_bar / self.T1
                c_ops.append(np.sqrt(gamma_up) * self.a_dag)
        
        if self.T_phi > 0:
            # Pure dephasing
            gamma_phi = 1.0 / self.T_phi
            c_ops.append(np.sqrt(gamma_phi) * self.n_op)
        
        return c_ops
    
    def _create_target_gate(self) -> np.ndarray:
        """Create target √X gate (R_X(π/2))"""
        sqrt_x = 0.5 * np.array([
            [1 + 1j, 1 - 1j],
            [1 - 1j, 1 + 1j]
        ], dtype=complex)
        return sqrt_x
    
    def _create_cardinal_states(self) -> List[qt.Qobj]:
        """
        Create six cardinal states:
        |0⟩, |1⟩, |+⟩, |-⟩, |+i⟩, |-i⟩
        """
        ket0 = qt.basis(self.n_levels, 0)
        ket1 = qt.basis(self.n_levels, 1)
        
        cardinal = [
            ket0,  # |0⟩
            ket1,  # |1⟩
            (ket0 + ket1).unit(),  # |+⟩ = (|0⟩ + |1⟩)/√2
            (ket0 - ket1).unit(),  # |-⟩ = (|0⟩ - |1⟩)/√2
            (ket0 + 1j*ket1).unit(),  # |+i⟩ = (|0⟩ + i|1⟩)/√2
            (ket0 - 1j*ket1).unit(),  # |-i⟩ = (|0⟩ - i|1⟩)/√2
        ]
        
        return cardinal
    
    def create_hamiltonian(self, pulses: np.ndarray, phi: float = 0.0, 
                          delta_freq: float = 0.0) -> List:
        """
        Create time-dependent Hamiltonian for QuTiP
        
        Parameters:
        -----------
        pulses : np.ndarray, shape (n_steps, 2)
            Pulse amplitudes [omega_I, omega_Q] in rad/s
        phi : float
            Phase rotation parameter
        delta_freq : float
            Frequency drift for this shot (Hz), converted to rad/s internally
        
        Returns:
        --------
        H : list
            QuTiP time-dependent Hamiltonian
        """
        omega_I = pulses[:, 0]
        omega_Q = pulses[:, 1]
        times = np.arange(len(pulses)) * self.dt
        
        # Apply phase rotation
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        omega_tilde_I = omega_I * cos_phi + omega_Q * sin_phi
        omega_tilde_Q = -omega_I * sin_phi + omega_Q * cos_phi
        
        # Create interpolation functions
        times_extended = np.append(times, times[-1] + self.dt)
        
        omega_I_func = interp1d(times_extended, 
                                np.append(omega_tilde_I, omega_tilde_I[-1]),
                                kind='previous', bounds_error=False, fill_value=0.0)
        omega_Q_func = interp1d(times_extended,
                                np.append(omega_tilde_Q, omega_tilde_Q[-1]),
                                kind='previous', bounds_error=False, fill_value=0.0)
        
        delta = self.delta
        
        # Coefficient functions
        def coeff_I_plus(t, args):
            return 0.5 * omega_I_func(t) * np.exp(-1j * delta * t)
        
        def coeff_I_minus(t, args):
            return 0.5 * omega_I_func(t) * np.exp(1j * delta * t)
        
        def coeff_Q_plus(t, args):
            return 0.5j * omega_Q_func(t) * np.exp(-1j * delta * t)
        
        def coeff_Q_minus(t, args):
            return -0.5j * omega_Q_func(t) * np.exp(1j * delta * t)
        
        # Static Hamiltonian with frequency drift for this shot
        # delta_freq is in Hz, convert to rad/s
        H_static = self.H_static_base
        if delta_freq != 0.0:
            H_static = H_static + 2 * np.pi * delta_freq * self.n_op
        
        # Construct Hamiltonian
        H = [
            H_static,
            [self.a_dag, coeff_I_plus],
            [self.a, coeff_I_minus],
            [self.a_dag, coeff_Q_plus],
            [self.a, coeff_Q_minus]
        ]
        
        return H
    
    def simulate_evolution(self, pulses: np.ndarray, phi: float, 
                          initial_state: qt.Qobj, delta_freq: float = 0.0) -> qt.Qobj:
        """
        Simulate open system evolution for a given initial state
        
        Parameters:
        -----------
        pulses : np.ndarray
            Pulse amplitudes
        phi : float
            Phase parameter
        initial_state : qt.Qobj
            Initial quantum state (ket)
        delta_freq : float
            Frequency drift for this shot (Hz)
        
        Returns:
        --------
        rho_final : qt.Qobj
            Final density matrix
        """
        H = self.create_hamiltonian(pulses, phi, delta_freq)
        times = np.arange(len(pulses) + 1) * self.dt
        
        # Convert to density matrix
        rho0 = qt.ket2dm(initial_state)
        
        # Evolve
        result = qt.mesolve(H, rho0, times, self.c_ops, [])
        
        return result.states[-1]
    
    def simulate_evolution_ensemble(self, pulses: np.ndarray, phi: float,
                                   initial_state: qt.Qobj, n_shots: int = None,
                                   seed: int = None) -> qt.Qobj:
        """
        Simulate evolution over an ensemble of shots with frequency drift noise
        
        For each shot, δu_f is sampled from N(0, σ_f²), representing shot-to-shot
        fluctuations in frequency. The final density matrix is the ensemble average.
        
        FIXED: Uses independent RandomState for better random number management
        
        Parameters:
        -----------
        pulses : np.ndarray
            Pulse amplitudes
        phi : float
            Phase parameter
        initial_state : qt.Qobj
            Initial quantum state (ket)
        n_shots : int or None
            Number of shots (if None, use self.n_shots)
        seed : int or None
            Random seed for reproducibility
        
        Returns:
        --------
        rho_avg : qt.Qobj
            Ensemble-averaged final density matrix (normalized)
        """
        if n_shots is None:
            n_shots = self.n_shots
        
        # FIXED: Use independent RandomState instead of global seed
        rng = np.random.RandomState(seed)
        
        # Storage for density matrices
        rho_ensemble = []
        
        for shot in range(n_shots):
            # Sample frequency drift: δu_f ~ N(0, σ_f²)
            delta_freq = rng.normal(0, self.sigma_freq)
            
            # Simulate this shot
            rho_final = self.simulate_evolution(pulses, phi, initial_state, delta_freq)
            rho_ensemble.append(rho_final)
        
        # Ensemble average: ⟨ρ⟩ = (1/N) Σ ρ_i
        rho_avg = sum(rho_ensemble) / n_shots
        
        # Check trace but don't force normalization (for consistency with v6)
        trace = rho_avg.tr()
        if abs(trace - 1.0) > 1e-3:
            print(f"Warning: Density matrix trace = {trace:.8f}, significant deviation!")
        
        # Optional: uncomment to force normalization (may differ from v6)
        # rho_avg = rho_avg / trace
        
        return rho_avg


    def evolution(self, pulses: np.ndarray, phi: float, n_shots: int = None,
                                   seed: int = None) -> List[qt.Qobj]:
        """
        计算所有 cardinal states 的演化结果（多进程优化）
        
        优化点：将6次串行计算改为并行计算
        """
        if n_shots is None:
            n_shots = self.n_shots

        # 准备参数
        params = [(self, pulses, phi, psi_i, n_shots, seed) for psi_i in self.cardinal_states]
        
        
        # 使用多进程并行计算
        with Pool() as pool:
            rho_final_avgs = pool.map(worker, params)
        
        return rho_final_avgs

    def compute_gate_error(self, pulses: np.ndarray, phi: float = 0.0,
                          n_shots: int = None, seed: int = None) -> Tuple[float, List[float]]:
        """
        Compute gate error ε_g using six cardinal states with ensemble averaging
        
        Formula:
        ε_g ≈ 1 - (1/6) Σ_i ⟨ψ_i| R_X(π/2)† ⟨ρ(ψ_i)⟩ R_X(π/2) |ψ_i⟩
        
        where ⟨ρ(ψ_i)⟩ is the ensemble-averaged density matrix over shots
        
        Parameters:
        -----------
        pulses : np.ndarray
            Pulse amplitudes
        phi : float
            Phase parameter
        n_shots : int or None
            Number of shots for ensemble averaging
        seed : int or None
            Random seed for reproducibility
        
        Returns:
        --------
        epsilon_g : float
            Gate error
        fidelities : list
            Individual state fidelities
        """
        if n_shots is None:
            n_shots = self.n_shots
        
        # Create target gate operator (2×2)
        R_x = self.target_gate_2x2
        R_x_dag = R_x.conj().T
        
        # Embed in full Hilbert space
        R_x_full_np = np.eye(self.n_levels, dtype=complex)
        R_x_full_np[:2, :2] = R_x
        R_x_full = qt.Qobj(R_x_full_np)
        
        R_x_dag_full_np = np.eye(self.n_levels, dtype=complex)
        R_x_dag_full_np[:2, :2] = R_x_dag
        R_x_dag_full = qt.Qobj(R_x_dag_full_np)
        
        fidelities = []
        
        for psi_i in self.cardinal_states:
            # Simulate evolution with ensemble averaging
            rho_final_avg = self.simulate_evolution_ensemble(pulses, phi, psi_i, 
                                                            n_shots, seed)
            
            # Apply gate correction: R_X† ⟨ρ⟩ R_X
            rho_corrected = R_x_dag_full * rho_final_avg * R_x_full
            
            # Compute fidelity: ⟨ψ_i| ρ_corrected |ψ_i⟩
            fidelity = qt.expect(qt.ket2dm(psi_i), rho_corrected)
            fidelities.append(np.real(fidelity))

        # Average fidelity
        avg_fidelity = np.mean(fidelities)
        
        # Gate error
        epsilon_g = 1.0 - avg_fidelity
        
        return epsilon_g, fidelities


    def compute_gate_error_2(self, rho_final_avg_list) -> Tuple[float, List[float]]:


        # Create target gate operator (2×2)
        R_x = self.target_gate_2x2
        R_x_dag = R_x.conj().T
        
        # Embed in full Hilbert space
        R_x_full_np = np.eye(self.n_levels, dtype=complex)
        R_x_full_np[:2, :2] = R_x
        R_x_full = qt.Qobj(R_x_full_np)
        
        R_x_dag_full_np = np.eye(self.n_levels, dtype=complex)
        R_x_dag_full_np[:2, :2] = R_x_dag
        R_x_dag_full = qt.Qobj(R_x_dag_full_np)
        
        fidelities = []

        for i in range(len(self.cardinal_states)):
            psi_i = self.cardinal_states[i]
            rho_final_avg = rho_final_avg_list[i]

            # Apply gate correction: R_X† ⟨ρ⟩ R_X
            rho_corrected = R_x_dag_full * rho_final_avg * R_x_full
            
            # Compute fidelity: ⟨ψ_i| ρ_corrected |ψ_i⟩
            fidelity = qt.expect(qt.ket2dm(psi_i), rho_corrected)
            fidelities.append(np.real(fidelity))
        
        # Average fidelity
        avg_fidelity = np.mean(fidelities)
        
        # Gate error
        epsilon_g = 1.0 - avg_fidelity
        
        return epsilon_g, fidelities

    def compute_leakage(self, pulses: np.ndarray, phi: float = 0.0,
                       n_shots: int = None, seed: int = None) -> Tuple[float, List[float]]:
        """
        Compute leakage L from computational subspace with ensemble averaging
        
        Formula:
        L ≈ (1/6) Σ {1 - Tr[|0⟩⟨0| ⟨ρ(ψ_i)⟩] - Tr[|1⟩⟨1| ⟨ρ(ψ_i)⟩]}
        
        where ⟨ρ(ψ_i)⟩ is the ensemble-averaged density matrix over shots
        
        Parameters:
        -----------
        pulses : np.ndarray
            Pulse amplitudes
        phi : float
            Phase parameter
        n_shots : int or None
            Number of shots for ensemble averaging
        seed : int or None
            Random seed for reproducibility
        
        Returns:
        --------
        leakage : float
            Average leakage probability
        individual_leakages : list
            Leakage for each cardinal state
        """
        if n_shots is None:
            n_shots = self.n_shots
        
        # Projectors
        proj_0 = qt.ket2dm(qt.basis(self.n_levels, 0))
        proj_1 = qt.ket2dm(qt.basis(self.n_levels, 1))
        
        leakages = []
        
        for psi_i in self.cardinal_states:
            # Simulate evolution with ensemble averaging
            rho_final_avg = self.simulate_evolution_ensemble(pulses, phi, psi_i, 
                                                            n_shots, seed)
            
            # Population in computational subspace
            pop_0 = np.real((proj_0 * rho_final_avg).tr())
            pop_1 = np.real((proj_1 * rho_final_avg).tr())
            
            # Leakage = 1 - (pop_0 + pop_1)
            leakage_i = 1.0 - pop_0 - pop_1
            leakages.append(leakage_i)
        # Average leakage
        avg_leakage = np.mean(leakages)
        
        return avg_leakage, leakages
    

    def compute_leakage_2(self, rho_final_avg_list) -> Tuple[float, List[float]]:

        # Projectors
        proj_0 = qt.ket2dm(qt.basis(self.n_levels, 0))
        proj_1 = qt.ket2dm(qt.basis(self.n_levels, 1))
        
        leakages = []
        
        for i in range(len(self.cardinal_states)):
            rho_final_avg = rho_final_avg_list[i]
            
            # Population in computational subspace
            pop_0 = np.real((proj_0 * rho_final_avg).tr())
            pop_1 = np.real((proj_1 * rho_final_avg).tr())
            
            # Leakage = 1 - (pop_0 + pop_1)
            leakage_i = 1.0 - pop_0 - pop_1
            leakages.append(leakage_i)


        # Average leakage
        avg_leakage = np.mean(leakages)
        
        return avg_leakage, leakages


    def compute_amplitude_penalty(self, pulses: np.ndarray) -> float:
        """
        Compute amplitude penalty P_a

        Formula:
        P_a = (A_a / N) × Σ_{j,k} [exp((u_{c,j}^{(k)} / h_a)^2) - 1]

        Parameters:
        -----------
        pulses : np.ndarray, shape (n_steps, 2)
            Pulse amplitudes in rad/s

        Returns:
        --------
        P_a : float
            Amplitude penalty
        """
        N = len(pulses)

        penalty = 0.0
        for j in range(N):
            for k in range(2):  # I and Q
                u_c = pulses[j, k]
                ratio_sq = (u_c / self.h_a)**2
                # Prevent numerical overflow (exp(50) ≈ 5e21)
                penalty += np.exp(min(ratio_sq, 50.0)) - 1
        
        P_a = (self.A_penalty / N) * penalty
        
        return P_a
    
    def compute_derivative_penalty(self, pulses: np.ndarray) -> float:
        """
        Compute derivative penalty P_d

        Formula:
        P_d = (A_d / N) × Σ_{j,k} [exp(((u_{c,j+1}^{(k)} - u_{c,j}^{(k)}) / h_d)^2) - 1]

        Parameters:
        -----------
        pulses : np.ndarray, shape (n_steps, 2)
            Pulse amplitudes in rad/s

        Returns:
        --------
        P_d : float
            Derivative penalty
        """
        N = len(pulses)

        penalty = 0.0
        for j in range(N - 1):
            for k in range(2):  # I and Q
                diff = pulses[j+1, k] - pulses[j, k]
                ratio_sq = (diff / self.h_d)**2
                # Prevent numerical overflow
                penalty += np.exp(min(ratio_sq, 50.0)) - 1
        
        P_d = (self.A_penalty / N) * penalty
        
        return P_d
    
    def grade_submission(self, pulses: np.ndarray, phi: float = 0.0, 
                        n_shots: int = None, seed: int = None,
                        verbose: bool = True) -> Dict:
        """
        Grade a pulse submission with all criteria
        
        Parameters:
        -----------
        pulses : np.ndarray, shape (n_steps, 2)
            Pulse amplitudes in rad/s [omega_I, omega_Q]
        phi : float
            Phase parameter
        n_shots : int or None
            Number of shots for ensemble averaging (if None, use self.n_shots)
        seed : int or None
            Random seed for reproducibility
        verbose : bool
            Print detailed results
        
        Returns:
        --------
        results : dict
            Grading results including all metrics and scores
        """
        # Validate input
        if pulses.ndim != 2 or pulses.shape[1] != 2:
            raise ValueError(f"Pulses must have shape (n_steps, 2), got {pulses.shape}")
        
        # Check if pulse length matches expected n_steps
        if len(pulses) != self.n_steps:
            print(f"Warning: Pulse has {len(pulses)} steps, expected {self.n_steps} steps")
        
        if n_shots is None:
            n_shots = self.n_shots
        
        if verbose:
            print(f"\nSimulating with {n_shots} shots for ensemble averaging...")
        
        # 串行运算
        if self.computing_method == 'serial':
            # 1. Gate error (primary metric) - with ensemble averaging
            epsilon_g, fidelities = self.compute_gate_error(pulses, phi, n_shots, seed)

            # 2. Leakage (secondary metric) - with ensemble averaging
            leakage, individual_leakages = self.compute_leakage(pulses, phi, n_shots, seed)

        elif self.computing_method == 'parallel':
            # 并行运算
            rho_final_avg_list = self.evolution(pulses, phi, n_shots, seed)
            epsilon_g, fidelities = self.compute_gate_error_2(rho_final_avg_list)
            leakage, individual_leakages = self.compute_leakage_2(rho_final_avg_list)

        

        # 3. Amplitude penalty
        P_a = self.compute_amplitude_penalty(pulses)
        
        # 4. Derivative penalty
        P_d = self.compute_derivative_penalty(pulses)
        
        # Compute scores
        # Primary score: based on gate fidelity (1 - epsilon_g)
        gate_fidelity = 1.0 - epsilon_g
        
        # Total penalty
        total_penalty = P_a + P_d
        
        # Overall score (weighted combination)
        # Gate fidelity: 80%, Leakage: 15%, Penalties: 5%
        leakage_score = max(0, 1 - leakage * 5)
        penalty_score = max(0, 1 - total_penalty)
        
        overall_score = (
            0.80 * gate_fidelity +
            0.15 * leakage_score +
            0.05 * penalty_score
        )
        
        results = {
            'gate_error': epsilon_g,
            'gate_fidelity': gate_fidelity,
            'individual_fidelities': fidelities,
            'leakage': leakage,
            'individual_leakages': individual_leakages,
            'amplitude_penalty': P_a,
            'derivative_penalty': P_d,
            'total_penalty': total_penalty,
            'leakage_score': leakage_score,
            'penalty_score': penalty_score,
            'overall_score': overall_score,
            'pulse_duration_ns': len(pulses) * self.dt * 1e9,
            'n_steps': len(pulses),
            'n_shots': n_shots,
            'sigma_freq_mhz': self.sigma_freq / 1e6,
            'h_a_hz': self.h_a_hz,
            'h_d_hz': self.h_d_hz
        }
        
        if verbose:
            self.print_results(results)
        
        return results
    
    def print_results(self, results: Dict):
        """Print grading results in a formatted way"""
        print("\n" + "="*70)
        print("PULSE GRADING RESULTS")
        print("="*70)
        
        print(f"\n{'PRIMARY METRICS':-^70}")
        print(f"Gate Error (ε_g):           {results['gate_error']:.6f}")
        print(f"Gate Fidelity (1-ε_g):      {results['gate_fidelity']:.6f} ({results['gate_fidelity']*100:.4f}%)")
        print(f"Leakage (L):                {results['leakage']:.6f} ({results['leakage']*100:.4f}%)")
        
        print(f"\n{'PULSE QUALITY METRICS':-^70}")
        print(f"Amplitude Penalty (P_a):    {results['amplitude_penalty']:.6f}")
        print(f"Derivative Penalty (P_d):   {results['derivative_penalty']:.6f}")
        print(f"Total Penalty:              {results['total_penalty']:.6f}")
        
        print(f"\n{'SCORING COMPONENTS':-^70}")
        print(f"Gate Fidelity Score (80%):  {results['gate_fidelity']:.6f}")
        print(f"Leakage Score (15%):        {results['leakage_score']:.6f}")
        print(f"Penalty Score (5%):         {results['penalty_score']:.6f}")
        
        print(f"\n{'INDIVIDUAL STATE FIDELITIES':-^70}")
        state_names = ['|0⟩', '|1⟩', '|+⟩', '|-⟩', '|+i⟩', '|-i⟩']
        for name, fid in zip(state_names, results['individual_fidelities']):
            print(f"  {name:6s}: {fid:.6f}")
        
        print(f"\n{'INDIVIDUAL STATE LEAKAGES':-^70}")
        for name, leak in zip(state_names, results['individual_leakages']):
            print(f"  {name:6s}: {leak:.6f}")
        
        print(f"\n{'PULSE INFORMATION':-^70}")
        print(f"Number of steps:            {results['n_steps']}")
        print(f"Pulse duration:             {results['pulse_duration_ns']:.2f} ns")
        print(f"Number of shots:            {results['n_shots']}")
        print(f"Frequency noise (σ_f):      {results['sigma_freq_mhz']:.3f} MHz")
        print(f"Amplitude threshold (h_a):  {results['h_a_hz']:.1f} Hz")
        print(f"Derivative threshold (h_d): {results['h_d_hz']:.1f} Hz")
        
        print(f"\n{'OVERALL SCORE':-^70}")
        print(f"Overall Score:              {results['overall_score']:.6f} ({results['overall_score']*100:.2f}%)")
        print("="*70 + "\n")
    
    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file"""
        # Convert numpy types to native Python types
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.float64, np.float32)):
                json_results[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                json_results[key] = int(value)
            elif isinstance(value, list):
                json_results[key] = [float(x) if isinstance(x, (np.float64, np.float32)) else x 
                                    for x in value]
            else:
                json_results[key] = value
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {filename}")


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    # Initialize grader with default parameters
    grader = TransmonPulseGrader(
        n_levels=4,
        n_steps=30,  # Expected number of time steps
        alpha=-2 * np.pi * 0.2e9,
        omega_q=2 * np.pi * 5.0e9,
        omega_d=2 * np.pi * 5.0e9,
        dt=5e-10,
        T1=50e-6,
        T_phi=30e-6,
        n_bar=0.05,
        sigma_freq=0.5e6,  # 0.5 MHz std for frequency drift
        n_shots=15,  # 15 shots for ensemble averaging (default)
        h_a=179e6,  # Will be converted to rad/s internally
        h_d=22.4e6,  # Will be converted to rad/s internally
        A_penalty=0.1
    )
    
    # Example: Load participant's pulse (replace with actual loading)
    # For demonstration, create a simple pulse
    n_steps = 30
    np.random.seed(42)
    
    # Simple Gaussian pulse envelope
    t = np.arange(n_steps)
    envelope = np.exp(-(t - n_steps/2)**2 / (2 * (n_steps/5)**2))
    
    amp = 2 * np.pi * 80e6  # 80 MHz amplitude
    pulses = np.column_stack([
        amp * envelope,
        amp * envelope * 0.5
    ])
    
    phi = 0.0
    
    # Grade the submission (use grader's default n_shots)
    print("\nGrading example pulse submission...")
    results = grader.grade_submission(pulses, phi, seed=42, verbose=True)
    
    # Save results
    grader.save_results(results, 'grading_results.json')
    
    print("\n" + "="*70)
    print("GRADER READY FOR HACKATHON SUBMISSIONS")
    print("="*70)
    print("\nTo grade a submission:")
    print("1. Load pulse data: pulses = np.load('submission.npy')")
    print("2. Grade: results = grader.grade_submission(pulses, phi)")
    print("3. Results will be printed and can be saved to JSON")
    print("\nScoring breakdown:")
    print("  - Gate Fidelity: 80%")
    print("  - Leakage:       15%")
    print("  - Penalties:      5%")
    print("="*70)
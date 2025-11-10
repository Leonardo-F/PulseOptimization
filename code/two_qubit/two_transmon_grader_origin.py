import numpy as np
import qutip as qt
from scipy.interpolate import interp1d
from typing import List, Tuple, Optional, Dict
import json


class DispersiveCNOTPulseGrader_Origin:
    """
    Grader for evaluating CNOT gate pulses in a dispersive-limit two-transmon system

    This grader simulates the pulse evolution using QuTiP with open system dynamics
    and evaluates the pulse quality based on multiple criteria.

    Physical Model (rotating frame at ω_d, RWA):
      H = H_0 + Ω_re(t) H_{d,re} + Ω_im(t) H_{d,im}
      
      H_0 = Σ_{q=1,2} [(ω_q - ω_d + α_q/2) b_q^† b_q - α_q/2 (b_q^† b_q)^2]
            + J (b_1^† b_2 + b_1 b_2^†)
      
      H_{d,re} = 1/2 [(b_1^† + b_1) + λ(b_2^† + b_2)]
      H_{d,im} = i/2 [(b_1^† - b_1) + λ(b_2^† - b_2)]

    Controls:
      - Ω_re(t), Ω_im(t): drive amplitudes in rad/s, piecewise-constant per time bin.

    Target:
      - Ideal two-qubit CNOT acting on {|00>,|01>,|10>,|11>} (qubit-1 control, qubit-2 target).

    Open System Dynamics:
      - T1/T_phi for each qubit
      - Shot-to-shot detuning noise (Gaussian) per qubit for ensemble averaging

    Evaluation Metrics:
      - Gate error ε_g via average state fidelity over 36 input states
      - Leakage from 2×2 computational subspace
      - Amplitude / derivative penalties for pulse quality constraints
    """

    def __init__(self,
                 # Hilbert space sizes
                 nq_levels: int = 3,
                 # Timing
                 n_steps: int = 300,
                 dt: float = 5e-10,            # 0.5 ns
                 # System params (GHz -> rad/s inside)
                 omega1_GHz: float = 4.380,    # ω_1/(2π) = 4.380 GHz
                 omega2_GHz: float = 4.614,    # ω_2/(2π) = 4.614 GHz
                 omega_d_GHz: float = 4.498,   # ω_d/(2π) = 4.498 GHz (drive frequency)
                 alpha1_GHz: float = 0.210,     # α_1/(2π) = 0.210 GHz (210 MHz)
                 alpha2_GHz: float = 0.215,     # α_2/(2π) = 0.215 GHz (215 MHz)
                 J_GHz: float = -0.003,         # J/(2π) = -0.003 GHz (-3 MHz)
                 lambda_coupling: float = 1.03, # λ: relative coupling strength
                 # Open system
                 T1_q1: float = 50e-6,
                 T1_q2: float = 50e-6,
                 Tphi_q1: float = 30e-6,
                 Tphi_q2: float = 30e-6,
                 nbar_q1: float = 0.0,
                 nbar_q2: float = 0.0,
                 # Shots
                 n_shots: int = 10,
                 sigma_detune_q1_Hz: float = 0.5e6,
                 sigma_detune_q2_Hz: float = 0.5e6,
                 # Penalties (Hz -> rad/s internally)
                 A_penalty: float = 0.1,
                 h_a_Hz: float = 200e6,
                 h_d_Hz: float = 2.7e6,
                 ):
        """
        Parameters:
        -----------
        nq_levels : int
            Number of transmon energy levels per qubit
        n_steps : int
            Expected number of time steps in pulse sequences
        dt : float
            Time step in seconds
        omega1_GHz, omega2_GHz : float
            Qubit transition frequencies in GHz
        omega_d_GHz : float
            Drive frequency (rotating frame reference) in GHz
        alpha1_GHz, alpha2_GHz : float
            Anharmonicity in GHz (typically positive in this convention)
        J_GHz : float
            Effective qubit-qubit coupling in GHz (typically negative)
        lambda_coupling : float
            Relative coupling strength λ for second qubit drive
        T1_q1, T1_q2 : float
            Energy relaxation time in seconds for each qubit
        Tphi_q1, Tphi_q2 : float
            Pure dephasing time in seconds for each qubit
        nbar_q1, nbar_q2 : float
            Thermal occupation number for each qubit
        n_shots : int
            Number of shots for ensemble averaging
        sigma_detune_q1_Hz, sigma_detune_q2_Hz : float
            Standard deviation of frequency drift noise in Hz per qubit
        A_penalty : float
            Penalty scaling factor
        h_a_Hz : float
            Amplitude threshold in Hz (will be converted to rad/s internally), default 200 MHz
        h_d_Hz : float
            Derivative threshold in Hz (will be converted to rad/s internally), default 2.7 MHz
        """
        self.nq = nq_levels
        self.n_steps = n_steps
        self.dt = dt
        self.times = np.arange(n_steps + 1) * dt

        # Convert to rad/s
        two_pi = 2 * np.pi
        self.omega1 = two_pi * omega1_GHz * 1e9
        self.omega2 = two_pi * omega2_GHz * 1e9
        self.omega_d = two_pi * omega_d_GHz * 1e9
        self.alpha1 = two_pi * alpha1_GHz * 1e9
        self.alpha2 = two_pi * alpha2_GHz * 1e9
        self.J = two_pi * J_GHz * 1e9
        self.lambda_coupling = lambda_coupling

        # Detunings from drive frequency
        self.delta1 = self.omega1 - self.omega_d  # ω_1 - ω_d
        self.delta2 = self.omega2 - self.omega_d  # ω_2 - ω_d

        # Open system params
        self.T1_q1 = T1_q1
        self.T1_q2 = T1_q2
        self.Tphi_q1 = Tphi_q1
        self.Tphi_q2 = Tphi_q2
        self.nbar_q1 = nbar_q1
        self.nbar_q2 = nbar_q2

        self.n_shots = n_shots
        self.sigma_detune_q1 = sigma_detune_q1_Hz
        self.sigma_detune_q2 = sigma_detune_q2_Hz

        # Penalties - Convert Hz to rad/s for unit consistency
        self.A_penalty = A_penalty
        self.h_a = two_pi * h_a_Hz
        self.h_d = two_pi * h_d_Hz

        # Store original values in Hz for reporting
        self.h_a_hz = h_a_Hz
        self.h_d_hz = h_d_Hz

        # Build operators
        # Tensor order: [qubit1, qubit2] (no resonator)
        b1 = qt.destroy(self.nq)
        b2 = qt.destroy(self.nq)
        Iq = qt.qeye(self.nq)

        self.b1 = qt.tensor(b1, Iq)
        self.b2 = qt.tensor(Iq, b2)

        self.b1_dag = self.b1.dag()
        self.b2_dag = self.b2.dag()

        self.n1 = self.b1_dag * self.b1
        self.n2 = self.b2_dag * self.b2

        # Anharmonicity terms: -α_q/2 (b_q^† b_q)^2
        # Note: (b^† b)^2 = b^† b^† b b + b^† b, so we use the simpler form
        self.H_anh = -(self.alpha1 / 2.0) * (self.n1 * self.n1) \
                   - (self.alpha2 / 2.0) * (self.n2 * self.n2)

        # Qubit-qubit coupling: J (b_1^† b_2 + b_1 b_2^†)
        self.H_coupling = self.J * (self.b1_dag * self.b2 + self.b1 * self.b2_dag)

        # Static drift: (ω_q - ω_d + α_q/2) b_q^† b_q
        self.H_drift_base = (self.delta1 + self.alpha1 / 2.0) * self.n1 \
                          + (self.delta2 + self.alpha2 / 2.0) * self.n2 \
                          + self.H_anh + self.H_coupling

        # Drive operators
        # H_{d,re} = 1/2 [(b_1^† + b_1) + λ(b_2^† + b_2)]
        # H_{d,im} = i/2 [(b_1^† - b_1) + λ(b_2^† - b_2)]
        self.H_drive_re = 0.5 * ((self.b1 + self.b1_dag) + 
                                 self.lambda_coupling * (self.b2 + self.b2_dag))
        self.H_drive_im = 0.5j * ((self.b1_dag - self.b1) + 
                                  self.lambda_coupling * (self.b2_dag - self.b2))

        # Collapse operators
        self.c_ops = self._setup_collapse_operators()

        # Target CNOT (q1 control -> q2 target)
        self.U_target = self._build_target_cnot()

        # Projector for leakage (2x2 subspace on qubits)
        self.P_comp = self._build_P_comp()


    # ---------- Basic blocks ----------

    def _setup_collapse_operators(self) -> List[qt.Qobj]:
        c_ops = []

        # Qubit 1
        if self.T1_q1 and self.T1_q1 > 0:
            gamma_down = (1 + self.nbar_q1) / self.T1_q1
            c_ops.append(np.sqrt(gamma_down) * self.b1)
            if self.nbar_q1 > 0:
                gamma_up = self.nbar_q1 / self.T1_q1
                c_ops.append(np.sqrt(gamma_up) * self.b1_dag)
        if self.Tphi_q1 and self.Tphi_q1 > 0:
            gamma_phi = 1.0 / self.Tphi_q1
            c_ops.append(np.sqrt(gamma_phi) * self.n1)

        # Qubit 2
        if self.T1_q2 and self.T1_q2 > 0:
            gamma_down = (1 + self.nbar_q2) / self.T1_q2
            c_ops.append(np.sqrt(gamma_down) * self.b2)
            if self.nbar_q2 > 0:
                gamma_up = self.nbar_q2 / self.T1_q2
                c_ops.append(np.sqrt(gamma_up) * self.b2_dag)
        if self.Tphi_q2 and self.Tphi_q2 > 0:
            gamma_phi = 1.0 / self.Tphi_q2
            c_ops.append(np.sqrt(gamma_phi) * self.n2)

        return c_ops

    def _build_target_cnot(self) -> qt.Qobj:
        # 4x4 CNOT on two qubits (q1 control, q2 target)
        CNOT = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [0,0,1,0]], dtype=complex)
        
        # Embed in nq x nq space - CORRECTED VERSION
        Uq = np.eye(self.nq * self.nq, dtype=complex)
        
        # Correct mapping: |00⟩=0, |01⟩=1, |10⟩=3, |11⟩=4 in 9x9 basis
        idx_map = [0, 1, self.nq, self.nq + 1]  # [0, 1, 3, 4]
        
        for i in range(4):
            for j in range(4):
                Uq[idx_map[i], idx_map[j]] = CNOT[i, j]
        
        U_target = qt.Qobj(Uq, dims=[[self.nq, self.nq], [self.nq, self.nq]])
        return U_target
   
    def _build_P_comp(self) -> qt.Qobj:
        # Projector onto computational subspace of both qubits (levels 0,1)
        Pq = 0 * qt.tensor(qt.qeye(self.nq), qt.qeye(self.nq))
        for i in [0, 1]:
            for j in [0, 1]:
                ket_ij = qt.tensor(qt.basis(self.nq, i), qt.basis(self.nq, j))
                Pq += qt.ket2dm(ket_ij)
        return Pq

    # ---------- Hamiltonian construction ----------

    def _build_time_dependent_H(self,
                                pulses: np.ndarray,
                                detune_offsets_Hz: Tuple[float, float] = (0.0, 0.0)) -> List:
        """
        pulses: shape (self.n_steps, 2) -> [Ω_re, Ω_im] in rad/s (PWC)
        """
        assert pulses.shape == (self.n_steps, 2), \
            f"Pulses must have shape ({self.n_steps}, 2), got {pulses.shape}"

        # Step-wise (hold) interpolation to piecewise-constant controls
        t_bins = np.arange(self.n_steps) * self.dt
        t_ext = np.append(t_bins, t_bins[-1] + self.dt)

        Omega_re, Omega_im = pulses[:, 0], pulses[:, 1]

        def step_interp(arr):
            return interp1d(t_ext, np.append(arr, arr[-1]),
                            kind='previous', bounds_error=False, fill_value=0.0)

        f_Omega_re = step_interp(Omega_re)
        f_Omega_im = step_interp(Omega_im)

        # Static detuning jitters (Hz -> rad/s)
        d1 = 2 * np.pi * detune_offsets_Hz[0]
        d2 = 2 * np.pi * detune_offsets_Hz[1]

        # Add jitters to H0
        H0 = self.H_drift_base + d1 * self.n1 + d2 * self.n2

        # Time-dependent Hamiltonian
        H = [
            H0,
            [self.H_drive_re, lambda t, args=None: float(f_Omega_re(t))],
            [self.H_drive_im, lambda t, args=None: float(f_Omega_im(t))],
        ]
        return H

    # ---------- Simulation ----------

    def _initial_kets_36(self) -> List[qt.Qobj]:
        # Single-qubit set: |0>, |1>, |+>, |->, |+i>, |-i>
        b0 = qt.basis(self.nq, 0)
        b1 = qt.basis(self.nq, 1)
        plus = (b0 + b1).unit()
        minus = (b0 - b1).unit()
        plus_i = (b0 + 1j * b1).unit()
        minus_i = (b0 - 1j * b1).unit()
        sq = [b0, b1, plus, minus, plus_i, minus_i]
        # Tensor products for two qubits
        states = []
        for s1 in sq:
            for s2 in sq:
                states.append(qt.tensor(s1, s2))
        return states
    
    def forward_propagation(self,pulses,state):
        H = self._build_time_dependent_H(pulses)
        rho0 = qt.ket2dm(state)
        result = qt.mesolve(H, rho0, self.times, self.c_ops, [])
        return result.states[-1]
    
    def simulate_one_shot(self,
                          pulses: np.ndarray,
                          detune_offsets_Hz: Tuple[float, float] = (0.0, 0.0)) -> List[qt.Qobj]:
        """
        Evolve the 36 input states; return list of final density matrices.
        """
        H = self._build_time_dependent_H(pulses, detune_offsets_Hz)
        rho_out_list = []
        for psi0 in self._initial_kets_36():
            rho0 = qt.ket2dm(psi0)
            result = qt.mesolve(H, rho0, self.times, self.c_ops, [])
            rho_out_list.append(result.states[-1])
        return rho_out_list

    def simulate_ensemble(self,
                          pulses: np.ndarray,
                          n_shots: Optional[int] = None,
                          seed: Optional[int] = None) -> List[qt.Qobj]:
        """
        Ensemble-average final states over detuning jitters on both qubits.
        Returns list of 36 density matrices.
        """
        if n_shots is None:
            n_shots = self.n_shots
        rng = np.random.RandomState(seed)

        accum = [0 for _ in range(36)]
        for s in range(n_shots):
            d1 = rng.normal(0.0, self.sigma_detune_q1)
            d2 = rng.normal(0.0, self.sigma_detune_q2)
            finals = self.simulate_one_shot(pulses, (d1, d2))
            if s == 0:
                accum = finals
            else:
                accum = [accum[i] + finals[i] for i in range(36)]
        avg = [rho / n_shots for rho in accum]
        return avg

    # ---------- Metrics ----------

    def gate_error(self,
                pulses: np.ndarray,
                n_shots: Optional[int] = None,
                seed: Optional[int] = None) -> Tuple[float, List[float]]:
        """
        Compute gate error ε_g using 36 input states with ensemble averaging
        """
        avg_states = self.simulate_ensemble(pulses, n_shots, seed)

        # Build same 36 inputs
        b0 = qt.basis(self.nq, 0)
        b1 = qt.basis(self.nq, 1)
        plus = (b0 + b1).unit()
        minus = (b0 - b1).unit()
        plus_i = (b0 + 1j * b1).unit()
        minus_i = (b0 - 1j * b1).unit()
        sq = [b0, b1, plus, minus, plus_i, minus_i]
        inputs = [qt.tensor(s1, s2) for s1 in sq for s2 in sq]

        fidelities = []
        for i in range(36):
            rho = avg_states[i]
            rho_corr = self.U_target.dag() * rho * self.U_target
            F_i = qt.expect(qt.ket2dm(inputs[i]), rho_corr)
            fidelities.append(float(np.real(F_i)))

        F_avg = float(np.mean(fidelities))
        eps_g = 1.0 - F_avg
        return eps_g, fidelities

    def leakage(self,
                pulses: np.ndarray,
                n_shots: Optional[int] = None,
                seed: Optional[int] = None) -> Tuple[float, List[float]]:
        """
        Compute leakage L from computational subspace with ensemble averaging
        """
        avg_states = self.simulate_ensemble(pulses, n_shots, seed)
        leaks = []
        for rho in avg_states:
            pop_comp = np.real((self.P_comp * rho).tr())
            leaks.append(float(1.0 - pop_comp))
        return float(np.mean(leaks)), leaks

    def amplitude_penalty(self, pulses: np.ndarray) -> float:
        """
        Compute amplitude penalty P_a for 2-channel pulses
        """
        N = pulses.shape[0]
        pen = 0.0
        for j in range(N):
            for k in range(2):  # Only 2 channels now
                r2 = (pulses[j, k] / self.h_a) ** 2
                pen += np.exp(min(r2, 50.0)) - 1.0
        return self.A_penalty * pen / N

    def derivative_penalty(self, pulses: np.ndarray) -> float:
        """
        Compute derivative penalty P_d for 2-channel pulses
        """
        N = pulses.shape[0]
        pen = 0.0
        for j in range(N - 1):
            for k in range(2):  # Only 2 channels now
                diff = pulses[j + 1, k] - pulses[j, k]
                r2 = (diff / self.h_d) ** 2
                pen += np.exp(min(r2, 50.0)) - 1.0
        return self.A_penalty * pen / N

    def grade_submission(self,
                        pulses: np.ndarray,
                        n_shots: Optional[int] = None,
                        seed: Optional[int] = None,
                        verbose: bool = True) -> Dict:
        """
        Grade a pulse submission with all criteria
        """
        # Validate input
        if pulses.ndim != 2 or pulses.shape[1] != 2:
            raise ValueError(f"Pulses must have shape (n_steps, 2), got {pulses.shape}")

        if len(pulses) != self.n_steps:
            print(f"Warning: Pulse has {len(pulses)} steps, expected {self.n_steps} steps")

        if n_shots is None:
            n_shots = self.n_shots

        if verbose:
            print(f"\nSimulating with {n_shots} shots for ensemble averaging...")

        # Compute metrics
        epsilon_g, individual_fidelities = self.gate_error(pulses, n_shots, seed)
        leakage, individual_leakages = self.leakage(pulses, n_shots, seed)
        P_a = self.amplitude_penalty(pulses)
        P_d = self.derivative_penalty(pulses)

        # Compute scores
        gate_fidelity = 1.0 - epsilon_g
        total_penalty = P_a + P_d
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
            'individual_fidelities': individual_fidelities,
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
            'sigma_detune_q1_mhz': self.sigma_detune_q1 / 1e6,
            'sigma_detune_q2_mhz': self.sigma_detune_q2 / 1e6,
            'h_a_hz': self.h_a_hz,
            'h_d_hz': self.h_d_hz
        }

        if verbose:
            self.print_results(results)

        return results

    def print_results(self, results: Dict):
        """Print grading results in a formatted way"""
        print("\n" + "="*70)
        print("PULSE GRADING RESULTS (Two-Qubit CNOT - Dispersive Limit)")
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

        print(f"\n{'INDIVIDUAL STATE FIDELITIES (36 input states)':-^70}")
        state_labels = ['0', '1', '+', '-', '+i', '-i']
        print("  " + "".join([f"  {label:>4s}  " for label in state_labels]))
        for i in range(6):
            row_str = f"{state_labels[i]:>2s}"
            for j in range(6):
                idx = i * 6 + j
                row_str += f"  {results['individual_fidelities'][idx]:.4f}"
            print(row_str)

        print(f"\n{'PULSE INFORMATION':-^70}")
        print(f"Number of steps:            {results['n_steps']}")
        print(f"Pulse duration:             {results['pulse_duration_ns']:.2f} ns")
        print(f"Number of channels:         2 (Ω_re, Ω_im)")
        print(f"Number of shots:            {results['n_shots']}")
        print(f"Frequency noise Q1 (σ_f):   {results['sigma_detune_q1_mhz']:.3f} MHz")
        print(f"Frequency noise Q2 (σ_f):   {results['sigma_detune_q2_mhz']:.3f} MHz")

        print(f"\n{'OVERALL SCORE':-^70}")
        print(f"Overall Score:              {results['overall_score']:.6f} ({results['overall_score']*100:.2f}%)")
        print("="*70 + "\n")

    def save_results(self, results: Dict, filename: str):
        """Save results to JSON file"""
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                json_results[key] = float(value)
            elif isinstance(value, list):
                json_results[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x
                                    for x in value]
            else:
                json_results[key] = value

        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"Results saved to {filename}")



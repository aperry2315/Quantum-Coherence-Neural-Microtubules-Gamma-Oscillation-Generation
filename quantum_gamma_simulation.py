import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt

"""
Quantum Coherence in Neural Microtubules: Simulation Engine
Author: Anthony L. Perry (2025)
VERSION 10.0: "The Entrainment Protocol" (GUARANTEED)
Logic: The Microtubule acts as a Pacemaker (40 Hz).
       Coherence determines the precision of this Pacemaker.
       High Coherence = Precise Kicks = High Precision.
       Low Coherence = Jittery Kicks = Low Precision.
"""

# ==========================================
# 1. QUANTUM DYNAMICS
# ==========================================
class MicrotubuleBundle:
    def __init__(self, temp_k=310.0):
        self.temp_k = temp_k
        self.Tc = 12.0  
        self.T0 = 310.0 
        
        # Base Coherence Calculation
        if self.temp_k > (self.T0 + 10):
            self.mean_rho = 0.05 
        else:
            delta_T = max(0, self.temp_k - self.T0)
            self.mean_rho = 0.95 * np.exp(-delta_T / self.Tc)
            
    def get_coherence_trace(self, duration_ms, dt):
        steps = int(duration_ms / dt)
        rho = np.zeros(steps)
        rho[0] = self.mean_rho
        
        # Very stable quantum state
        tau_corr = 50.0 
        sigma = 0.002   
        
        noise = np.random.normal(0, 1, steps)
        for i in range(1, steps):
            drift = -(rho[i-1] - self.mean_rho) / tau_corr
            diffusion = sigma * np.sqrt(dt) * noise[i]
            rho[i] = rho[i-1] + drift * dt + diffusion
            
        return np.clip(rho, 0.0, 1.0)

# ==========================================
# 2. NEURAL NETWORK
# ==========================================
class PINGNetwork:
    def __init__(self, n_exc=400, n_inh=100, dt=0.1):
        self.Ne = n_exc
        self.Ni = n_inh
        self.dt = dt
        
        # Neuron Physics
        self.Cm = 0.5    
        self.gl = 0.025  
        self.El = -70.0  
        self.Vt = -50.0  
        self.Vr = -60.0  
        self.Ref = 2.0   
        
        # Time Constants
        self.tau_ampa = 2.0
        self.tau_gaba = 4.0  
        
        # Connectivity
        self.p_conn = 0.1
        self.Wee = 0.02 * (np.random.rand(self.Ne, self.Ne) < self.p_conn)
        self.Wei = 0.05 * (np.random.rand(self.Ni, self.Ne) < self.p_conn)
        self.Wie = 0.10 * (np.random.rand(self.Ne, self.Ni) < self.p_conn) 
        self.Wii = 0.05 * (np.random.rand(self.Ni, self.Ni) < self.p_conn)
        
    def run(self, duration_ms, mt_bundle):
        steps = int(duration_ms / self.dt)
        rho_t = mt_bundle.get_coherence_trace(duration_ms, self.dt)
        
        Ve = np.ones(self.Ne) * self.El + np.random.rand(self.Ne) * 10
        Vi = np.ones(self.Ni) * self.El + np.random.rand(self.Ni) * 10
        
        s_ampa = np.zeros(self.Ne) 
        s_gaba = np.zeros(self.Ni) 
        
        ref_e = np.zeros(self.Ne)
        ref_i = np.zeros(self.Ni)
        
        lfp_trace = [] 
        
        # PACEMAKER SETUP
        gamma_freq = 40.0 # Hz
        period_ms = 1000.0 / gamma_freq # 25ms
        next_kick_time = 5.0 # Start time
        
        print(f"Running: T={mt_bundle.temp_k:.1f}K, Rho={mt_bundle.mean_rho:.2f}")
        
        for t_idx in range(steps):
            current_time = t_idx * self.dt
            current_rho = rho_t[t_idx]
            
            # 1. PACEMAKER LOGIC
            # If it's time for a kick, we deliver current
            I_pacemaker = 0.0
            
            # Check if we crossed the kick time
            if current_time >= next_kick_time:
                # Deliver Kick! (Short pulse)
                I_pacemaker = 10.0 
                
                # Calculate NEXT kick time
                # Ideally: current + 25ms
                # Reality: current + 25ms + Jitter
                
                # Jitter Calculation:
                # Rho=1.0 -> Jitter = 0.5 ms (Very Precise)
                # Rho=0.0 -> Jitter = 15.0 ms (Very Sloppy)
                jitter_sigma = 15.0 * (1.0 - 0.95 * current_rho)
                
                actual_jitter = np.random.normal(0, jitter_sigma)
                next_kick_time = current_time + period_ms + actual_jitter
            
            # Decay pacemaker current (pulse width ~1ms)
            # Simplified: We just add it for one step. 
            
            # 2. INPUT CURRENTS
            # Base drive is low, relying on Pacemaker to sync
            I_ext_e = 1.5 + np.random.normal(0, 0.5, self.Ne) + I_pacemaker
            I_ext_i = 1.0 + np.random.normal(0, 0.2, self.Ni) + (I_pacemaker * 0.5)
            
            # Synaptic Currents
            I_ampa_e = np.dot(self.Wee, s_ampa) * (Ve - 0)
            I_gaba_e = np.dot(self.Wie, s_gaba) * (Ve + 75)
            
            I_ampa_i = np.dot(self.Wei, s_ampa) * (Vi - 0)
            I_gaba_i = np.dot(self.Wii, s_gaba) * (Vi + 75)
            
            # UPDATE VOLTAGES
            dVe = (1/self.Cm) * (self.gl*(self.El - Ve) - I_ampa_e - I_gaba_e + I_ext_e)
            Ve_new = Ve + dVe * self.dt
            Ve_new[ref_e > 0] = self.Vr
            ref_e[ref_e > 0] -= self.dt
            
            dVi = (1/self.Cm) * (self.gl*(self.El - Vi) - I_ampa_i - I_gaba_i + I_ext_i)
            Vi_new = Vi + dVi * self.dt
            Vi_new[ref_i > 0] = self.Vr
            ref_i[ref_i > 0] -= self.dt
            
            # SPIKES
            spikes_e = Ve_new > self.Vt
            spikes_i = Vi_new > self.Vt
            
            Ve_new[spikes_e] = self.Vr
            Vi_new[spikes_i] = self.Vr
            ref_e[spikes_e] = self.Ref
            ref_i[spikes_i] = self.Ref
            
            # SYNAPSES
            s_ampa += (-s_ampa / self.tau_ampa) * self.dt + spikes_e.astype(float)
            s_gaba += (-s_gaba / self.tau_gaba) * self.dt + spikes_i.astype(float)
            
            Ve = Ve_new
            Vi = Vi_new
            
            # LFP
            lfp_val = np.mean(np.abs(I_ampa_e) + np.abs(I_gaba_e))
            lfp_trace.append(lfp_val)
            
        return np.array(lfp_trace), rho_t

# ==========================================
# 3. ANALYSIS
# ==========================================
def analyze_precision(lfp, dt):
    fs = 1000 / dt
    nyq = 0.5 * fs
    b, a = butter(4, [30/nyq, 100/nyq], btype='band')
    
    startup_idx = int(200/dt)
    lfp_clean = lfp[startup_idx:] if len(lfp) > startup_idx else lfp
    gamma_lfp = filtfilt(b, a, lfp_clean)
    
    peaks = []
    # Strict Threshold
    threshold = np.mean(gamma_lfp) + 1.0 * np.std(gamma_lfp)
    
    for i in range(1, len(gamma_lfp)-1):
        if gamma_lfp[i] > gamma_lfp[i-1] and gamma_lfp[i] > gamma_lfp[i+1]:
            if gamma_lfp[i] > threshold: 
                peaks.append(i * dt)
    
    if len(peaks) > 3:
        isis = np.diff(peaks)
        jitter = np.std(isis)
        # Standard Precision Metric
        precision = 1.0 / (jitter + 0.01)
    else:
        precision = 0.02 # Floor
        
    return precision

# ==========================================
# 4. MAIN RUNNER
# ==========================================
if __name__ == "__main__":
    
    temps = np.linspace(308, 320, 15) 
    precisions = []
    coherences = []
    
    print("Starting Entrainment Protocol Simulation...")
    
    for T in temps:
        mt = MicrotubuleBundle(temp_k=T)
        net = PINGNetwork()
        lfp, rho_trace = net.run(duration_ms=1000, mt_bundle=mt)
        
        prec = analyze_precision(lfp, 0.1)
        mean_rho = np.mean(rho_trace)
        
        precisions.append(prec)
        coherences.append(mean_rho)
        print(f" -> T={T:.1f} | Rho={mean_rho:.2f} | Precision={prec:.3f}")

    # Plot
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(coherences, precisions, c=temps, cmap='coolwarm', s=120, edgecolor='k', zorder=2)
    
    # Fit
    if len(coherences) > 1:
        z = np.polyfit(coherences, precisions, 1)
        p = np.poly1d(z)
        plt.plot(coherences, p(coherences), "r--", linewidth=2.5, zorder=1, label=f"Fit (Slope={z[0]:.2f})")
    
    cbar = plt.colorbar(sc)
    cbar.set_label('Temperature (K)')
    plt.xlabel('Microtubule Coherence Factor (rho)')
    plt.ylabel('Gamma Precision (1/std_ISI)')
    plt.title('Simulation Results: Coherence vs Gamma Precision')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('gamma_correlation_plot.png')
    print("\nSimulation Complete. Image saved.")

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback

# ==============================================================================
# SCRIPT OTTIMIZZATO PER GOOGLE COLAB - TRAINING INTENSIVO (100.000 STEP)
# ==============================================================================

class FairCityEnv(gym.Env):
    def __init__(self, data_path="smart_city_traffic_mobility.csv"):
        super(FairCityEnv, self).__init__()
        
        print("Caricamento Dataset Completo in RAM (204.000 righe)...")
        self.full_df = pd.read_csv(data_path)
        self.timestamps = self.full_df['timestamp'].unique()
        
        # Build Adjacency Graph once
        coords_df = self.full_df.drop_duplicates(subset=['intersection_id'])[['latitude', 'longitude']].values
        tri = Delaunay(coords_df)
        diff = coords_df[:, np.newaxis, :] - coords_df[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=-1))
        
        self.adj_list = {i: set() for i in range(len(coords_df))}
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    u, v = simplex[i], simplex[j]
                    if dist[u, v] < 0.04:
                        self.adj_list[u].add(v)
                        self.adj_list[v].add(u)
                        
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(4,), dtype=np.float32)
        
        self.max_steps = 1
        self.global_emission_cap = 500.0
        self.current_df = None
        self.current_ts = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        self.current_ts = np.random.choice(self.timestamps)
        self.current_df = self.full_df[self.full_df['timestamp'] == self.current_ts].copy().reset_index(drop=True)
        obs = self._get_obs(self.current_df)
        return obs, {}
        
    def step(self, action):
        self.current_step += 1
        env_w, modal_w, spatial_w = action
        
        agg_df = self.current_df.copy()
        agg_df['simulated_vehicle_count'] = agg_df['vehicle_count'].astype(float)
        agg_df['simulated_heavy_vehicle_count'] = agg_df['heavy_vehicle_count'].astype(float)
        
        for _ in range(3):
            vehicle_delta = np.zeros(len(agg_df))
            heavy_delta = np.zeros(len(agg_df))
            
            for i in range(len(agg_df)):
                neighbors = list(self.adj_list.get(i, []))
                if not neighbors: continue
                    
                cur_heavy = agg_df.at[i, 'simulated_heavy_vehicle_count']
                cur_veh = agg_df.at[i, 'simulated_vehicle_count']
                is_vuln = (agg_df.at[i, 'nearby_school'] == 1) or (agg_df.at[i, 'nearby_hospital'] == 1)
                
                n_veh_avg = np.mean([agg_df.at[n, 'simulated_vehicle_count'] for n in neighbors])
                total_lanes = sum([agg_df.at[n, 'lanes'] for n in neighbors])
                if total_lanes == 0: continue
                
                if agg_df.at[i, 'emission_estimate'] > self.global_emission_cap and cur_veh > 0:
                    to_dump = cur_veh * 0.5
                    vehicle_delta[i] -= to_dump
                    for n in neighbors: vehicle_delta[n] += to_dump * (agg_df.at[n, 'lanes'] / total_lanes)
                
                if is_vuln and cur_heavy > 0:
                    to_move = cur_heavy * (env_w * 0.8)
                    heavy_delta[i] -= to_move
                    for n in neighbors: heavy_delta[n] += to_move * (agg_df.at[n, 'lanes'] / total_lanes)
                        
                diff_veh = cur_veh - n_veh_avg
                if diff_veh > 0:
                    to_flow = diff_veh * (spatial_w * 0.5)
                    vehicle_delta[i] -= to_flow
                    for n in neighbors: vehicle_delta[n] += to_flow * (agg_df.at[n, 'lanes'] / total_lanes)
                        
            agg_df['simulated_vehicle_count'] = np.maximum(agg_df['simulated_vehicle_count'] + vehicle_delta, 0)
            agg_df['simulated_heavy_vehicle_count'] = np.maximum(agg_df['simulated_heavy_vehicle_count'] + heavy_delta, 0)
            
        ratio_veh = np.where(agg_df['vehicle_count'] > 0, agg_df['simulated_vehicle_count'] / agg_df['vehicle_count'], 1.0)
        modal_shift = 1.0 - (agg_df['public_transport_count'] / (agg_df['public_transport_count'].max() + 1)) * modal_w
        
        def hcm_delay(q_flow, lanes):
            sat_flow = lanes * 1800
            Y_base = np.clip(q_flow / sat_flow, 0.01, 0.95)
            C = (1.5 * 10.0 + 5) / (1 - Y_base)
            g = np.maximum(15.0, (C - 10.0) / 2.0)
            lam = np.clip(g / C, 0.05, 0.95)
            q_flow = np.maximum(1.0, q_flow)
            capacity = lam * sat_flow
            x = q_flow / np.maximum(capacity, 1.0)
            denom1 = 2 * (1.0 - np.minimum(lam * x, 0.99))
            d_uniform = (C * ((1.0 - lam) ** 2)) / np.maximum(denom1, 0.01)
            q_sec = q_flow / 3600.0
            d_random = np.where(x < 0.95, (x ** 2) / np.maximum(2 * q_sec * (1.0 - x), 0.001), 15.0 + 120.0 * (x - 0.95))
            return np.clip(d_uniform + d_random, 2.0, 900.0)

        base_hcm_delay = hcm_delay(agg_df['vehicle_count'], agg_df['lanes'])
        sim_hcm_delay = hcm_delay(agg_df['simulated_vehicle_count'], agg_df['lanes'])
        delay_ratio = np.where(base_hcm_delay > 0, sim_hcm_delay / base_hcm_delay, 1.0)
        
        agg_df['simulated_wait'] = agg_df['average_wait_time'] * delay_ratio * modal_shift
        agg_df['simulated_emission'] = agg_df['emission_estimate'] * (ratio_veh ** 2)
        
        obs = self._get_obs(agg_df)
        max_vuln_emission, wait_var, avg_bus_wait, global_avg_wait = obs
        
        # Calculate total city emissions (kg) and maximum wait time
        total_city_emission = agg_df['simulated_emission'].sum()
        max_city_wait = agg_df['simulated_wait'].max()
        
        # Calculate total city emissions (kg) and maximum wait time
        total_city_emission = agg_df['simulated_emission'].sum()
        max_city_wait = agg_df['simulated_wait'].max()
        
        # OTTIMIZZAZIONE "FAIRNESS-FIRST" (Per vincere la sfida sui 3 Goal)
        
        # 1. Environmental Justice (Priorita' ASSOLUTA)
        r_env = - (max_vuln_emission * 5.0)           
        
        # 2. Spatial Equity (Bilanciamento dei quartieri)
        r_spa = - (wait_var * 5.0)                    
        
        # 3. Modal Equality (Protezione totale dei Bus)
        r_mod = - (avg_bus_wait * 5.0)                
        
        # Obiettivi Secondari (Mantenere la citta' funzionale)
        r_glo = - (global_avg_wait * 1.0)             # Il ritardo globale pesato a 1
        r_total_emission = - (total_city_emission * 0.01)
        r_max_delay = - (max_city_wait * 0.5)
        
        reward = r_env + r_total_emission + r_spa + r_mod + r_glo + r_max_delay
        
        done = True
        return obs, reward, done, False, {}

    def _get_obs(self, df):
        is_vuln = (df['nearby_school'] == 1) | (df['nearby_hospital'] == 1)
        e_col = 'simulated_emission' if 'simulated_emission' in df.columns else 'emission_estimate'
        w_col = 'simulated_wait' if 'simulated_wait' in df.columns else 'average_wait_time'
        
        max_vuln_emission = df[is_vuln][e_col].max() if len(df[is_vuln]) > 0 else 0
        wait_var = df[w_col].var()
        total_buses = df['public_transport_count'].sum()
        avg_bus_wait = (df[w_col] * df['public_transport_count']).sum() / total_buses if total_buses > 0 else df[w_col].mean()
        global_avg_wait = df[w_col].mean()
        
        return np.array([max_vuln_emission, wait_var, avg_bus_wait, global_avg_wait], dtype=np.float32)

if __name__ == "__main__":
    print("Inizializzazione FairCityEnv...")
    env = FairCityEnv()
    
    print("Creazione Modello PPO...")
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=0.0003, n_steps=1024, batch_size=64, ent_coef=0.01)
    
    print("Inizio Addestramento (10,000 Step)...")
    model.learn(total_timesteps=10000, progress_bar=True)
    
    print("Salvataggio modello in corso...")
    model.save("fair_city_rl_model_converged")
    print("Modello 'fair_city_rl_model_converged.zip' salvato con successo!")

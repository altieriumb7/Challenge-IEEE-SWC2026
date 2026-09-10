import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from train_rl import FairCityEnv

def evaluate_performance():
    print("Caricamento Ambiente e Modello AI...")
    env = FairCityEnv()
    model = PPO.load("fair_city_rl_model.zip")
    
    # Test su 10 ore storiche
    np.random.seed(42)
    test_timestamps = np.random.choice(env.timestamps, size=10, replace=False)
    
    baseline_rewards = []
    ai_rewards = []
    
    for ts in test_timestamps:
        env.reset() # Inizializza le variabili di base (come current_step)
        
        env.current_ts = ts
        env.current_df = env.full_df[env.full_df['timestamp'] == ts].copy().reset_index(drop=True)
        obs = env._get_obs(env.current_df)
        
        # 1. BASELINE: Il Sindaco umano non fa nulla (tutto a zero)
        _, reward_base, _, _, _ = env.step([0.0, 0.0, 0.0])
        baseline_rewards.append(reward_base)
        
        # 2. AI OPTIMIZED: Il Modello RL agisce
        # Resettiamo l'ambiente allo stesso timestamp per un confronto alla pari
        env.reset()
        env.current_df = env.full_df[env.full_df['timestamp'] == ts].copy().reset_index(drop=True)
        obs = env._get_obs(env.current_df)
        action, _ = model.predict(obs, deterministic=True)
        _, reward_ai, _, _, _ = env.step(action)
        ai_rewards.append(reward_ai)
        
    avg_base = np.mean(baseline_rewards)
    avg_ai = np.mean(ai_rewards)
    
    print("\n" + "="*50)
    print("[REPORT PERFORMANCE MODELLO AI (Su 10 Scenari)]")
    print("="*50)
    print(f"Penalita' Media Baseline (Nessun Intervento): {avg_base:.2f}")
    print(f"Penalita' Media AI (PPO Fast-Train):        {avg_ai:.2f}")
    
    if avg_ai > avg_base: # Ricorda: i Reward sono negativi, quindi più vicino a 0 è meglio
        miglioramento = ((avg_base - avg_ai) / avg_base) * 100
        print(f"\n[+] L'Intelligenza Artificiale ha MIGLIORATO il sistema del {miglioramento:.2f}%")
    else:
        print("\n[!] Il modello ha bisogno di piu' addestramento (Miglioramento assente).")
        
if __name__ == "__main__":
    evaluate_performance()

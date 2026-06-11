import numpy as np
import joblib
import torch
import torcs_jm_par as snakeoil

# ==========================================
# IMPORTAZIONE DELLA RETE ESTERNA
# ==========================================
from train_modello_bc import TorcsNet 

# ==========================================
# 1. UTILITY
# ==========================================
def normalize_track(track) -> list:
    """Garantisce che i sensori della pista siano sempre 19 e formattati correttamente."""
    if not isinstance(track, (list, tuple, np.ndarray)):
        return [200.0] * 19
    track = list(track)
    if len(track) < 19:
        track += [200.0] * (19 - len(track))
    return track[:19]

def shift_gears(S):
    """Semplice logica euristica per il cambio marcia, dato che la rete non lo prevede."""
    gear = 1
    # VALORI MODIFICATI: Soglie allungate prese dal setup di qualifica ibrido
    speeds = [0, 45, 65, 110, 150, 200]
    for i, speed in enumerate(speeds):
        if S['speedX'] > speed:
            gear = i + 1
    return min(gear, 6)

# ==========================================
# 2. CARICAMENTO MODELLO E SCALER
# ==========================================
INPUT_DIM = 22 # speedX, trackPos, angle + 19 sensori

model = TorcsNet(INPUT_DIM)
payload = torch.load('modello_torcs_pytorch.pth', map_location='cpu')

if isinstance(payload, dict) and 'model_state_dict' in payload:
    model.load_state_dict(payload['model_state_dict'])
else:
    model.load_state_dict(payload)

model.eval() 
scaler = joblib.load('scaler_torcs.pkl')

# ==========================================
# 3. IL CERVELLO DELL'AI (Guida Pura)
# ==========================================
def drive_ai(c):
    S = c.S.d
    R = c.R.d
    
    track = normalize_track(S.get('track', []))
    X_input = [S.get('speedX', 0.0), S.get('trackPos', 0.0), S.get('angle', 0.0)] + track
    X_input_array = np.array([X_input])
    
    X_input_scaled = scaler.transform(X_input_array)
    X_tensor = torch.tensor(X_input_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        predizioni = model(X_tensor).numpy()[0]
    
    # VALORI MODIFICATI: Aggiunto un moltiplicatore 1.5 allo sterzo 
    R['steer'] = np.clip(predizioni[0] * 1.5, -1.0, 1.0)
    R['accel'] = np.clip(predizioni[1], 0.0, 1.0)
    R['brake'] = np.clip(predizioni[2], 0.0, 1.0)
    
    R['gear'] = shift_gears(S)
    
    return

# ==========================================
# ESECUZIONE MAIN LOOP
# ==========================================
if __name__ == "__main__":
    C = snakeoil.Client() 
    print("Client TORCS avviato. L'Agente Neurale è al volante (Pure BC).")
    
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        drive_ai(C) 
        C.respond_to_server()
        
    C.shutdown()
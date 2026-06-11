import glob
import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import joblib

# Import di PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- MODIFICA EFFETTUATA QUI ---
CARTELLA_CAMPIONI = 'giri_frenata'
FEATURES = ['speedX', 'trackPos', 'angle'] + [f'track_{i}' for i in range(19)]
TARGETS = ['steer', 'accel', 'brake']

def find_data_folder() -> str:
    if os.path.isdir(CARTELLA_CAMPIONI):
        return CARTELLA_CAMPIONI
    raise FileNotFoundError(f"Nessuna cartella '{CARTELLA_CAMPIONI}' trovata nella directory corrente.")

def load_data(folder: str) -> pd.DataFrame:
    # --- MODIFICA EFFETTUATA QUI: Cerca solo i file giro_fr_*.csv ---
    pattern = os.path.join(folder, 'giro_fr_*.csv')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"Nessun file nel formato 'giro_fr_xx.csv' trovato in '{folder}'.")

    dfs = [pd.read_csv(path) for path in files]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Trovati {len(files)} file CSV. Totale campioni: {len(df)}")
    return df

# ==========================================
# DEFINIZIONE DELLA RETE NEURALE PYTORCH (MIGLIORATA)
# ==========================================
class TorcsNet(nn.Module):
    def __init__(self, input_size):
        super(TorcsNet, self).__init__()
        
        # 3 strati comuni per l'estrazione delle feature geometriche e cinematiche
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),  # Previene l'overfitting spegnendo casualmente il 10% dei neuroni
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Diramazioni indipendenti (Teste di Output separate)
        self.steer_head = nn.Linear(32, 1)
        self.accel_head = nn.Linear(32, 1)
        self.brake_head = nn.Linear(32, 1)

    def forward(self, x):
        # Passaggio attraverso la spina dorsale comune
        features = self.shared_layers(x)
        
        # Calcolo dei singoli output con funzioni di attivazione specifiche:
        # Lo sterzo deve variare tra -1.0 (tutto a sinistra) e +1.0 (tutto a destra) -> Tanh
        steer = torch.tanh(self.steer_head(features))
        
        # L'acceleratore deve variare rigorosamente tra 0.0 e 1.0 -> Sigmoid
        accel = torch.sigmoid(self.accel_head(features))
        
        # Il freno deve variare rigorosamente tra 0.0 e 1.0 -> Sigmoid
        brake = torch.sigmoid(self.brake_head(features))
        
        # Concateniamo i tre output lungo la dimensione delle feature [batch_size, 3]
        return torch.cat((steer, accel, brake), dim=1)

# ==========================================
# ESECUZIONE PRINCIPALE
# ==========================================
if __name__ == "__main__":
    # Gestione del dispositivo di calcolo: seleziona CUDA (GPU) se disponibile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo hardware selezionato per l'addestramento: {device}")

    # 1. Carica e prepara i dati
    cartella = find_data_folder()
    df = load_data(cartella)
    
    X = df[FEATURES].values
    y = df[TARGETS].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inizializzazione e calcolo del normalizzatore
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. Conversione dati in TENSORI di PyTorch
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Creazione del DataLoader per l'addestramento a batch
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # 3. Inizializzazione Rete, Loss e Ottimizzatore
    input_dim = len(FEATURES) # 22
    
    # Creiamo il modello e lo spostiamo sul device (GPU o CPU)
    model = TorcsNet(input_dim).to(device)
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. CICLO DI ADDESTRAMENTO (Training Loop)
    epochs = 50
    print("\nInizio addestramento su hardware dedicato...")
    
    for epoch in range(epochs):
        model.train() # Attiva modalità addestramento (abilita Dropout)
        epoch_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            # SPOSTAMENTO DEI DATI DEL BATCH SULLA GPU/CPU
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()               # 1. Azzera i gradienti precedenti
            outputs = model(batch_X)            # 2. Forward pass (previsione)
            loss = criterion(outputs, batch_y)  # 3. Calcola l'errore combinato
            loss.backward()                     # 4. Calcola i gradienti (Backpropagation)
            optimizer.step()                    # 5. Aggiorna i pesi della rete
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoca [{epoch+1}/{epochs}], Loss Media: {epoch_loss/len(train_loader):.4f}")

    # 5. Valutazione sul Test Set
    model.eval() # Disattiva il Dropout per la fase di test
    with torch.no_grad(): # Disabilita il calcolo dei gradienti per risparmiare VRAM
        # Spostiamo l'intero test set sul device per l'inferenza rapida
        X_test_tensor = X_test_tensor.to(device)
        
        # Otteniamo le predizioni e le riportiamo sulla CPU convertendole in array Numpy
        test_predictions = model(X_test_tensor).cpu().numpy()
        
        # Calcolo delle metriche di accuratezza complessive e specifiche
        r2_total = r2_score(y_test, test_predictions)
        print(f"\nAddestramento completato!\nAccuratezza Complessiva (R^2 Score): {r2_total:.4f}")
        
        print("\nAnalisi di accuratezza per singolo controllo:")
        for i, target_name in enumerate(TARGETS):
            r2_single = r2_score(y_test[:, i], test_predictions[:, i])
            print(f" - R^2 Score per '{target_name}': {r2_single:.4f}")

    # 6. Salvataggio
    joblib.dump(scaler, 'scaler_torcs.pkl')
    
    # Riportiamo lo state_dict sulla CPU prima di salvarlo. 
    # Questo trucco evita problemi se caricherai il file in futuro su un PC senza GPU.
    metadata = {
        'framework': 'pytorch',
        'saved_at': datetime.now().isoformat(),
        'model_state_dict': model.cpu().state_dict(),
        'input_dim': input_dim,
        'output_dim': len(TARGETS),
        'feature_names': FEATURES,
        'target_names': TARGETS,
        'epochs': epochs,
        'optimizer': 'adam',
        'scaler_filename': 'scaler_torcs.pkl',
    }
    torch.save(metadata, 'modello_torcs_pytorch.pth')
    print("\nModello esportato con successo in 'modello_torcs_pytorch.pth'!")
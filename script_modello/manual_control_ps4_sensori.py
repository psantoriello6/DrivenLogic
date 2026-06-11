import pygame
import snakeoil3_jm2 as snakeoil3
import time
import json

class ArcadeController:
    def __init__(self):
        # Inizializza pygame e i joystick
        pygame.init()
        pygame.joystick.init()
        
        self.last_shift_time = 0  # <--- NUOVO: Timer per evitare cambiate multiple
        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1
        }

        # Controlla se ci sono controller collegati
        if pygame.joystick.get_count() == 0:
            print("NESSUN CONTROLLER RILEVATO! Collega il controller PS4 e riavvia.")
            self.joystick = None
        else:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            print(f"Controller rilevato: {self.joystick.get_name()}")

        # ==========================================
        # MAPPATURA ASSI (DualShock 4 Standard)
        # ==========================================
        self.AXIS_STEER = 0  # Analogico sinistro (sinistra/destra)
        self.AXIS_L2 = 4     # Grilletto Freno
        self.AXIS_R2 = 5     # Grilletto Acceleratore

    def update(self, sensors):
        # Fondamentale per aggiornare gli input del controller ad ogni frame
        pygame.event.pump() 

        speed = sensors.get('speedX', 0)
        angle = sensors.get('angle', 0)
        rpm = sensors.get('rpm', 0)

        raw_steer, raw_l2, raw_r2 = 0.0, -1.0, -1.0

        if self.joystick:
            raw_steer = -self.joystick.get_axis(self.AXIS_STEER)
            raw_l2 = self.joystick.get_axis(self.AXIS_L2)
            raw_r2 = self.joystick.get_axis(self.AXIS_R2)

        # I grilletti L2/R2 in Pygame restituiscono un valore da -1.0 a 1.0. Li normalizziamo da 0.0 a 1.0
        accel_input = ((raw_r2 + 1.0) / 2.0) * 1.2  # <--- MODIFICA: Accelerazione amplificata
        brake_input = (raw_l2 + 1.0) / 2.0

        current_time = time.time()
        shift_cooldown = 0.3 
        can_shift = (current_time - self.last_shift_time) > shift_cooldown

        # ========================
        # GESTIONE RETROMARCIA (Stile Arcade)
        # ========================
        if brake_input > 0.5 and speed < 2.0 and self.state['gear'] != -1 and can_shift:
            self.state['gear'] = -1
            self.last_shift_time = current_time
            
        elif self.state['gear'] == -1 and accel_input > 0.1 and can_shift:
            self.state['gear'] = 1
            self.last_shift_time = current_time

        # INVERSIONE PEDALI IN RETROMARCIA
        if self.state['gear'] == -1:
            target_accel = brake_input
            target_brake = accel_input
        else:
            target_accel = accel_input
            target_brake = brake_input

        # ========================
        # ACCELERAZIONE & FRENO (Con Smoothing)
        # ========================
        self.state['accel'] += (target_accel - self.state['accel']) * 0.2
        self.state['brake'] += (target_brake - self.state['brake']) * 0.3

        # ========================
        # CAMBIO AUTOMATICO (Scalata Dinamica + Protezione Marce Basse)
        # ========================
        if self.state['gear'] > 0 and can_shift:
            
            # UPSHIFT: Aumenta marcia a 15.000 RPM
            if rpm > 15000 and self.state['gear'] < 6:
                self.state['gear'] += 1
                self.last_shift_time = current_time
            
            # DOWNSHIFT LOGIC:
            else:
                if accel_input < 0.1 and brake_input < 0.1:
                    downshift_limit = 11500 
                elif brake_input > 0.1:
                    downshift_limit = 10000 
                else:
                    downshift_limit = 9500 

                if self.state['gear'] == 3:
                    downshift_limit = 8000
                elif self.state['gear'] == 2:
                    downshift_limit = 5000

                if rpm < downshift_limit and self.state['gear'] > 1:
                    self.state['gear'] -= 1
                    self.last_shift_time = current_time

        # ========================
        # STEERING INPUT
        # ========================
        if abs(raw_steer) < 0.05:
            raw_steer = 0.0
        else:
            raw_steer = raw_steer * 1.4  

        max_steer = max(0.25, 1.0 - speed / 200.0)
        steer_input = raw_steer * max_steer

        if abs(steer_input) < 0.01:
            steer_target = 0.0
        else:
            stability = angle * 0.3
            steer_target = steer_input - stability

        self.state['steer'] += (steer_target - self.state['steer']) * 0.2

        # CLAMP FINALE
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['accel'] = max(0.0, min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0, min(1.0, self.state['brake']))
        self.state['gear'] = max(-1, min(6, self.state['gear']))

# ============================================================
# MAIN
# ============================================================

def main():
    # vision=True serve al server se vuoi che calcoli immagini, ma per i laser (track) basta False.
    # Assicurati solo che in TORCS i laser siano abilitati.
    client = snakeoil3.Client(p=3001, vision=False)
    controller = ArcadeController()
    client.get_servers_input()

    print("\n--- Modalità Guida Controller PS4 Attiva ---")
    print("Levetta Sinistra: Sterza | R2: Accelera | L2: Frena")
    print("Premi il tasto OPTIONS sul pad o Ctrl+C nel terminale per terminare il giro.\n")

    # --- MODIFICA 1: Aggiornata l'intestazione del CSV per i 19 sensori ---
    track_headers = ",".join([f"track_{i}" for i in range(19)])
    log_csv_lines = [f"time,steer,accel,brake,gear,speedX,trackPos,angle,rpm,damage,{track_headers}\n"]
    
    log_json_data = []
    
    t0 = time.time()
    step = 0
    last_log_time = 0
    LOG_INTERVAL = 0.1 

    last_cur_lap_time = 0

    try:
        while True:
            S = client.S.d
            controller.update(S)
            a = controller.state
            
            if step % 500 == 0:
                print(f"[{step}] steer={a['steer']:.2f} accel={a['accel']:.2f} brake={a['brake']:.2f} gear={a['gear']}")

            client.R.d['steer'] = a['steer']
            client.R.d['accel'] = a['accel']
            client.R.d['brake'] = a['brake']
            client.R.d['gear'] = a['gear']
            client.R.d['clutch'] = 0.0
            client.R.d['meta'] = 0

            client.respond_to_server()
            client.get_servers_input()

            current_time = time.time() - t0

            # Salvataggio dati temporaneo
            if current_time - last_log_time >= LOG_INTERVAL:
                
                # --- MODIFICA 2: Estrazione sicura dei 19 sensori laser ---
                raw_track = S.get('track', [0.0]*19)
                if isinstance(raw_track, (list, tuple)):
                    track_floats = [float(v) for v in raw_track[:19]]
                else:
                    track_floats = [0.0] * 19
                    
                # Li uniamo in una stringa separata da virgole
                track_csv_str = ",".join([f"{v:.4f}" for v in track_floats])

                # --- MODIFICA 3: Salvataggio nel CSV ---
                log_csv_lines.append(
                    f"{current_time:.3f},{a['steer']},{a['accel']},{a['brake']},{a['gear']},"
                    f"{S.get('speedX',0)},{S.get('trackPos',0)},{S.get('angle',0)},"
                    f"{S.get('rpm',0)},{S.get('damage',0)},{track_csv_str}\n"
                )

                # --- MODIFICA 4: Salvataggio nel JSON ---
                log_json_data.append({
                    "step": step,
                    "time": round(current_time, 3),
                    "action": {"steer": a['steer'], "accel": a['accel'], "brake": a['brake'], "gear": a['gear']},
                    "state": {
                        "speedX": S.get('speedX', 0), 
                        "trackPos": S.get('trackPos', 0), 
                        "angle": S.get('angle', 0), 
                        "rpm": S.get('rpm', 0), 
                        "damage": S.get('damage', 0),
                        "track": track_floats  # Array dei laser
                    }
                })
                last_log_time = current_time

            # ==========================================
            # CONTROLLI DI FINE GIRO / INTERRUZIONE
            # ==========================================
            cur_lap_time = S.get('curLapTime', 0)
            dist_raced = S.get('distRaced', 0)

            if dist_raced > 500 and cur_lap_time < (last_cur_lap_time - 5):
                print(f"\n[TRAGUARDO SUPERATO!] Giro completato. Tempo a display: {last_cur_lap_time:.2f}s")
                break
                
            last_cur_lap_time = cur_lap_time

            if controller.joystick and controller.joystick.get_button(9):
                print("\n[STOP MANUALE] Registrazione interrotta dal controller.")
                break

            step += 1
            
    except KeyboardInterrupt:
        print("\n[STOP MANUALE] Interrotto tramite Ctrl+C.")
        
    finally:
        print("\n--- RILEVAZIONE TERMINATA ---")
        if pygame.get_init():
            pygame.quit() 
        
        scelta = input(f"Hai registrato {step} frame. Vuoi salvare i log di questo giro? (s/n): ").strip().lower()
        
        if scelta in ['s', 'si', 'y', 'yes']:
            print("Salvataggio dei file in corso, non chiudere la finestra...")
            with open("manual_log.csv", "w") as f_csv:
                f_csv.writelines(log_csv_lines)
            with open("manual_log.json", "w") as f_json:
                json.dump(log_json_data, f_json, indent=2)
            print("Salvataggio completato con successo!")
        else:
            print("Dati scartati.")

if __name__ == "__main__":
    main()
import cv2 
import numpy as np 
import tensorflow as tf  
from ultralytics import YOLO  
from collections import defaultdict  # dictionnaire avec valeurs par défaut


# 1 CHARGEMENT DES MODÈLES

yolo_model = YOLO("yolov8s.pt")  # modèle de détection + tracking
world_model = tf.keras.models.load_model("models/world_model.keras")  # modèle ML entraîné


# 2 CHARGEMENT DE LA VIDÉO

video_path = "data/videos/2.mp4"  # chemin de la vidéo
cap = cv2.VideoCapture(video_path)  # ouverture de la vidéo

# vérification que la vidéo s'ouvre bien
if not cap.isOpened():
    print("Erreur vidéo")
    exit()

# récupération de la taille originale de la vidéo
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# taille pour affichage uniquement (plus léger)
display_width = 640
display_height = 360

# facteurs pour convertir coordonnées originales → affichage
scale_disp_x = display_width / orig_width
scale_disp_y = display_height / orig_height

# Writer
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output2.mp4", fourcc, fps, (display_width, display_height)) # output...  numéro à changer

# 3 PARAMÈTRES

allowed_classes = {"car", "truck", "bus", "motorcycle", "person"} 

# couleurs pour chaque type d'objet
colors = {
    "car": (0,255,0),
    "truck": (0,0,255),
    "bus": (255,0,0),
    "motorcycle": (255,255,0),
    "person": (255,0,255)
}

sequence_length = 3  # taille de la séquence utilisée par le modèle
future_steps = 7  # nombre de prédictions futures

dt = 0.5  # intervalle de temps utilisé pendant le training 
#alpha = 0.8  # facteur de smoothing (réduction du bruit)

# historique des positions (pour affichage trajectoire)
track_history = defaultdict(list)

# états ML [x, y, vx, vy]
track_states = defaultdict(list)

frame_count = 0  # compteur de frames (pour optimiser les prédictions)


# 4 FONCTION DE PRÉDICTION 

def predict_future_fast(model, seq, steps=5):
    seq = seq.copy()  # copie de la séquence initiale
    preds = []  # liste des prédictions

    for _ in range(steps):  # boucle pour prédire plusieurs steps
        # inference rapide (beaucoup plus rapide que model.predict)
        pred = model(seq[np.newaxis], training=False).numpy()[0]

        preds.append(pred)  # stockage de la prédiction

        # mise à jour de la séquence (sliding window)
        seq = np.vstack([seq[1:], pred])

    return np.array(preds, dtype=np.float32)  # conversion en array

# 5 FONCTION DANGER
def compute_danger(x,y,vx,vy):
    score=0

      # Hauteur (adaptée au filtre 0.7)
    if y > 0.6:
        score+=3
    if y > 0.45:
        score+=2
    elif y > 0.3:
        score+=1
    
    # Largeur
    if 0.45 <= x <= 0.55:
        score+=2
    elif 0.3 <= x <= 0.7:
        score+=1

    # Mouvement
    if vy > 0.02:
        score+=3
    elif vy > 0.005:
        score+=1

    return score

# 6 Description

def describe_objects(x,y,vx,vy,name,danger):

    # Position horizontale
    if x < 0.4:
        pos_x="left"
    elif x > 0.6:
        pos_x="right"
    else:
        pos_x="center"

    # Distance
    if y > 0.6:
        distance="very close"
    elif y > 0.4:
        distance="close"
    else:
        distance="far"
    
    # Mouvement
    if vy > 0.02:
        motion="approaching fast"
    elif vy > 0.005:
        motion="approaching"
    else:
        motion="stable"

    return f"A {name} is {distance} ahead on the {pos_x}, {motion}"


# 7 BOUCLE PRINCIPALE

while True:

    ret, frame = cap.read()  # lecture d'une frame

    if not ret:  # fin de vidéo
        break

    frame_count += 1  # incrément du compteur

   
    # YOLO SUR IMAGE ORIGINALE 
  
    results = yolo_model.track(frame, persist=True, verbose=False)  # détection + tracking

    # image redimensionnée uniquement pour affichage
    display_frame = cv2.resize(frame, (display_width, display_height))

    # vérifie si des objets sont détectés
    if results[0].boxes.id is not None:

        boxes = results[0].boxes  # bounding boxes
        ids = boxes.id.int().cpu().tolist()  # IDs des objets

        # Paramètres danger maximum
        max_danger=-1 # -1 car inférieur à tous les scores possibles de 0 à ...
        most_dangerous=None

        # boucle sur chaque objet détecté
        for box, obj_id in zip(boxes, ids):

            cls = int(box.cls[0])  # classe de l'objet
            name = yolo_model.names[cls]  # nom de la classe

            # filtre des classes
            if name not in allowed_classes:
                continue

            # coordonnées bounding box (image originale)
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # centre de l'objet
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # ignore dashboard / ego car (bas de l'image)
            if cy > orig_height * 0.7: # Si cy supérieur à 70% de la hauteur de l'image (en bas), la zone est ignorée
                continue

            # conversion coordonnées → affichage
            x1_d = int(x1 * scale_disp_x)
            y1_d = int(y1 * scale_disp_y)
            x2_d = int(x2 * scale_disp_x)
            y2_d = int(y2 * scale_disp_y)

            cx_d = int(cx * scale_disp_x)
            cy_d = int(cy * scale_disp_y)

            color = colors[name]  # couleur associée

           
            # AFFICHAGE BOUNDING BOX
          
            cv2.rectangle(display_frame, (x1_d,y1_d), (x2_d,y2_d), color, 2)

            cv2.putText(display_frame, f"{name} {obj_id}",
                        (x1_d, y1_d-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2)

            
            # TRAJECTOIRE PASSÉE
            
            track = track_history[obj_id]  # historique de l'objet
            track.append((cx_d, cy_d))  # ajout du point

            if len(track) > 15:  # limite de taille
                track.pop(0)

            # dessin des lignes de trajectoire
            for i in range(1, len(track)):
                cv2.line(display_frame, track[i-1], track[i], color, 2)

            
            # NORMALISATION (POUR ML)
            
            x_norm = cx / orig_width
            y_norm = cy / orig_height

          
            # CALCUL VITESSE + SMOOTHING
            
            if len(track_states[obj_id]) > 0:

                prev_x, prev_y, _, _ = track_states[obj_id][-1]

                # vitesse brute (avant smoothing)
                raw_vx = (x_norm - prev_x) / dt
                raw_vy = (y_norm - prev_y) / dt

                speed = abs(raw_vx) + abs(raw_vy)

                # alpha dynamique
                if speed < 0.01:
                    alpha = 0.9   # très stable si objet lent
                else:
                    alpha = 0.6   # plus réactif si mouvement rapide

                # smoothing (réduit le bruit YOLO)
                x_norm = alpha * prev_x + (1 - alpha) * x_norm
                y_norm = alpha * prev_y + (1 - alpha) * y_norm

                # vitesse cohérente avec training
                vx = (x_norm - prev_x) / dt
                vy = (y_norm - prev_y) / dt

            else:
                vx = 0.0
                vy = 0.0

            # ajout de l'état
            track_states[obj_id].append([x_norm, y_norm, vx, vy])

            danger = compute_danger(x_norm,y_norm,vx,vy)

            if danger > max_danger:
                max_danger=danger
                most_dangerous = (x_norm,y_norm,vx,vy,name,danger)

            if danger>=6:
                danger_color=(0,0,255) # rouge
            elif danger>=3:
                danger_color=(0,165,255) # orange
            else:
                danger_color=(0,255,0) # vert 

            if len(track_states[obj_id]) > 30:
                track_states[obj_id].pop(0)

            # ignore objets trop loin
            if y_norm < 0.25:
                continue

            # Affiche le score de danger
            cv2.putText(display_frame, f"D:{danger}",
            (x1_d, y1_d - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            danger_color, 2)

            # PRÉDICTION FUTURE
           
            if len(track_states[obj_id]) >= sequence_length and frame_count % 3 == 0: # Si assez d'historique pour l'objet (3 séquences) et si frame de 3 (prédiction toutes les 3 frames)

                # récupération de la séquence récente
                seq = np.array(track_states[obj_id][-sequence_length:], dtype=np.float32)

                # prédiction future
                preds = predict_future_fast(world_model, seq, steps=future_steps)

                # Smoothing des prédictions
                preds = preds * 0.7 + seq[-1] * 0.3 # Stabiliser les prédictions

                pred_points = []

                # conversion prédictions → affichage
                for pred in preds:
                    px = int(pred[0] * display_width)
                    py = int(pred[1] * display_height)
                    pred_points.append((px, py))

                # dessin des trajectoires futures
                for i in range(1, len(pred_points)):
                    cv2.line(display_frame, pred_points[i-1], pred_points[i], (0,165,255), 2)

                # points futurs
                for pt in pred_points:
                    cv2.circle(display_frame, pt, 3, (0,165,255), -1)

        if most_dangerous is not None:
            x, y, vx, vy, name, danger = most_dangerous

            desc=describe_objects(x, y, vx, vy, name, danger) # Description image avec danger prioritaire

            text = desc

            # taille du texte
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            # rectangle noir derrière
            cv2.rectangle(display_frame, (50, 90 - text_h - 5), (50 + text_w, 90 + 5), (0,0,0), -1)

            # texte rouge
            cv2.putText(display_frame, text,
                        (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255,255,255), 2)

            #cv2.putText(display_frame, desc,
                        #(50, 90),
                        #cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #(255,255,255), 2)

            cv2.putText(display_frame, f"MAIN: {name} ({danger})",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,0,255), 3)   


    # Sauvegarde
    out.write(display_frame) # Ajoute le frame dans la video

    # affichage final
    cv2.imshow("World Model V4 (FULL RES YOLO)", display_frame)

    # quitter avec ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# libération ressources
cap.release()
out.release() # libère fichier (sauvegarde)
cv2.destroyAllWindows()
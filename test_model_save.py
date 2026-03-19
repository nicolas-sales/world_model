import cv2
from ultralytics import YOLO
from collections import defaultdict # dictionnaire spécial, crée automatiquement des listes
                                    # utile pour stocker les trajectoires

model = YOLO("yolov8s.pt")

video_path = "data/videos/8.mp4"
cap = cv2.VideoCapture(video_path) # donne une image (frame)

# writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v") # encodeur vidéo (format .mp4)
out = cv2.VideoWriter("output.mp4", fourcc, 30, (640, 360)) # crée un fichier vidéo output.mp4 avec fps=30 et la dimension

allowed_classes = {"car", "truck", "bus", "motorcycle", "person"}

colors = {
    "car": (0,255,0),
    "truck": (0,0,255),
    "bus": (255,0,0),
    "motorcycle": (255,255,0),
    "person": (255,0,255)
}

track_history = defaultdict(list) # pour chaque object_id, on stocke ses positions

while True: # boucle infinie, lit toute la vidéo
    
    ret, frame = cap.read() # frame : image, ret True ou False
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 360)) # réduction de taille : plus rapide
    
    results = model.track(frame, persist=True, verbose=False) # détecte + suit les objets. True pour garder les IDs entre frames
    
    if results[0].boxes.id is not None: # Vérifie si présence d'objets
        
        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist() # IDs (convertis en liste Python)
        
        for box, obj_id in zip(boxes, ids): # boucle sur chaque objet détecté
            
            cls = int(box.cls[0])
            name = model.names[cls] # transforme ID en nom
            
            if name not in allowed_classes: # ignore les objets inutiles
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0]) # bounding box, coin haut gauche, coin bas droite
            
            cx = int((x1 + x2) / 2) # Position de l'objet utilisé pour la trajectoire
            cy = int((y1 + y2) / 2)
            
            color = colors[name] # coulmeur selon le type
            
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2) # dessine rectangle
            
            cv2.putText(frame, f"{name} {obj_id}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) # Affiche le type et l'Id
            
            # Trajectoire
            track = track_history[obj_id]
            track.append((cx, cy)) # Ajoute la position actuelle
            
            if len(track) > 30:
                track.pop(0) # Garde seulement les 30 derniers points
            
            for i in range(1, len(track)):
                cv2.line(frame, track[i-1], track[i], color, 2) # Relie les points pour obtenir une trajectoire
    
    # sauvegarde
    out.write(frame) # Ajoite le frame dans la video
    
    cv2.imshow("Tracking filtré", frame) # Affiche le frame
    
    if cv2.waitKey(30) & 0xFF == 27: # Appuyer sur ESC pour arrêter
        break

cap.release() # libère video
out.release() # libère fichier
cv2.destroyAllWindows() # libère fenêtre
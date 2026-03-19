import cv2
from ultralytics import YOLO
from collections import defaultdict

# modèle
model = YOLO("yolov8s.pt")

video_path = "data/videos/12.mp4"
cap = cv2.VideoCapture(video_path)

# classes autorisées
allowed_classes = {"car", "truck", "bus", "motorcycle", "person"}

# couleurs par type
colors = {
    "car": (0,255,0),
    "truck": (0,0,255),
    "bus": (255,0,0),
    "motorcycle": (255,255,0),
    "person": (255,0,255)
}

# trajectoires
track_history = defaultdict(list)

while True:
    
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.resize(frame, (640, 360))
    
    results = model.track(frame, persist=True, verbose=False)
    
    if results[0].boxes.id is not None:
        
        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist()
        
        for box, obj_id in zip(boxes, ids):
            
            cls = int(box.cls[0])
            name = model.names[cls]
            
            # 🔥 filtre
            if name not in allowed_classes:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            color = colors[name]
            
            # bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            
            # label
            cv2.putText(
                frame,
                f"{name} {obj_id}",
                (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            
            # trajectoire
            track = track_history[obj_id]
            track.append((cx, cy))
            
            if len(track) > 30:
                track.pop(0)
            
            for i in range(1, len(track)):
                cv2.line(frame, track[i-1], track[i], color, 2)
    
    cv2.imshow("Tracking filtré", frame)
    
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
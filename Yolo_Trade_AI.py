import sys
import os
import threading
import numpy as np
import mss
import cv2
import time
from pynput import keyboard
from PyQt5 import QtWidgets, QtCore, QtGui
from ultralytics import YOLO

# === CONFIG ===
#MODEL_PATH = "model.pt"
#MODEL_PATH = r"C:\Users\otomy\Documents\GitHub\YOLO_test\dataset\runs\detect\train16\weights\best.pt"
MODEL_PATH = r"C:\Users\otomy\Documents\GitHub\YOLO_test\best_drink.pt"

#qCLASSES = ['Head and shoulders bottom', 'Head and shoulders top', 'M_Head', 'W_Bottom']
CLASSES = ['coffee', 'sugar', 'tea']
        
# === OVERLAY ===
class Overlay(QtWidgets.QWidget):
    def __init__(self, model, classes):
        super().__init__()
        self.model = model
        self.classes = classes
        self.boxes = []
        self.labels = []

        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)

        screen_geometry = QtWidgets.QApplication.primaryScreen().geometry()
        self.setGeometry(screen_geometry)

        # FPS tracking
        self.frame_times = []
        self.current_fps = 0

        # Timers
        self.repaint_timer = QtCore.QTimer()
        self.repaint_timer.timeout.connect(self.update)
        self.repaint_timer.start(30)

        self.detection_timer = QtCore.QTimer()
        self.detection_timer.timeout.connect(self.run_detection)
        self.detection_timer.start(0)

        print("[INFO] Detección iniciada...")

    def run_detection(self):
        start = time.time()
        
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        try:
            #t0 = time.time()
            results = self.model(img, verbose=False)
            #print(f"[DEBUG] inferencia tomó {time.time() - t0:.4f} segundos")

            self.boxes = []
            self.labels = []

            if results and results[0].boxes:
                for box in results[0].boxes:
                    coords = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = f"{self.classes[class_id]} ({conf:.2f})"

                    x1, y1, x2, y2 = map(int, coords)
                    self.boxes.append((x1, y1, x2, y2))
                    self.labels.append(label)


        except Exception as e:
            print(f"[ERROR] during detection or overlay update: {e}")
            self.boxes = []
            self.labels = []

        # FPS update
        end = time.time()
        self.frame_times.append(end)
        self.frame_times = [t for t in self.frame_times if end - t <= 1.0]
        self.current_fps = len(self.frame_times)

        #print(f"[DEBUG] Current FPS: {self.current_fps}")

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setFont(QtGui.QFont("Arial", 18, QtGui.QFont.Bold))

        # Permanent title and FPS counter
        painter.setPen(QtGui.QColor(255, 0, 0))
        painter.drawText(30, 50, "Overlay is active")
        painter.drawText(30, 80, f"FPS: {self.current_fps}")

        # Squares and labels
        for (x1, y1, x2, y2), label in zip(self.boxes, self.labels):
            rect = QtCore.QRect(x1, y1, x2 - x1, y2 - y1)

            painter.setPen(QtGui.QColor(255, 0, 0))  # rojo para el cuadro
            painter.drawRect(rect)

            painter.setPen(QtGui.QColor(0, 255, 0))  # verde para el texto
            painter.drawText(x1, y1 - 10, label)
            
# === Escape key ===
def start_keyboard_listener():
    def on_press(key):
        try:
            if key.char == 'q':
                print("[INFO] Escape key 'q' detected. Closing...")
                os._exit(0)
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

# === MAIN ===
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        sys.exit(1)

    print("[INFO] Loading model...")
    
    model = YOLO(MODEL_PATH)
    #model.to("cpu")
    model.to("cuda")
    
    print("[INFO] Model successfully loaded.")

    app = QtWidgets.QApplication(sys.argv)
    overlay = Overlay(model, CLASSES)
    overlay.show()

    threading.Thread(target=start_keyboard_listener, daemon=True).start()
    sys.exit(app.exec_())

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras.layers import Dropout
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import time
from concurrent.futures import ThreadPoolExecutor

# Configuration
CONFIG = {
    'model_path': r"C:\Users\piyan\Downloads\codes all\Deepfake\DeepFake-Detect-master\best_model.h5",
    'input_size': (128, 128),
    'threshold': 0.5,
    'history_file': 'detection_history.txt'
}

# Custom Dropout layer
class FixedDropout(Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=False)

get_custom_objects().update({'swish': tf.nn.swish, 'FixedDropout': FixedDropout})

class DeepfakeDetector:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            with tf.keras.utils.custom_object_scope({'FixedDropout': FixedDropout, 'swish': tf.nn.swish}):
                self.model = load_model(CONFIG['model_path'])
        except Exception as e:
            print(f"Error loading model: {e}")
            messagebox.showerror("Model Load Error", "Failed to load the model.")

    def preprocess_image(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read image")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, CONFIG['input_size'])
            image_normalized = image_resized / 255.0
            return np.expand_dims(image_normalized, axis=0), image_rgb
        except Exception as e:
            print(f"Preprocessing error: {e}")
            return None, None

    def predict(self, image_path):
        if self.model is None:
            return None
        image_batch, original = self.preprocess_image(image_path)
        if image_batch is None:
            return None
        prediction = self.model.predict(image_batch, verbose=0)[0][0]
        is_fake = prediction > CONFIG['threshold']
        self.save_history(image_path, is_fake, prediction)
        return {
            'filename': os.path.basename(image_path),
            'is_fake': is_fake,
            'confidence': round(prediction * 100, 2),
            'original_image': original
        }

    def save_history(self, image_path, is_fake, confidence):
        with open(CONFIG['history_file'], 'a') as file:
            file.write(f"{os.path.basename(image_path)}, {'FAKE' if is_fake else 'REAL'}, {confidence:.2f}%\n")

class DeepfakeDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DeepGuard - Deepfake Detection")
        self.root.geometry("950x750")
        self.root.configure(bg="#222831")
        self.detector = DeepfakeDetector()
        self.executor = ThreadPoolExecutor(max_workers=4)

        # UI Elements
        self.title_label = tk.Label(root, text="DeepGuard - Deepfake Detection", font=("Arial", 20, "bold"), fg="#00adb5", bg="#222831")
        self.title_label.pack(pady=10)

        self.button_frame = tk.Frame(root, bg="#222831")
        self.button_frame.pack(pady=10)

        self.upload_btn = self.create_button(self.button_frame, "Upload Image", self.load_image, 0)
        self.history_btn = self.create_button(self.button_frame, "View History", self.view_history, 1)
        self.reset_btn = self.create_button(self.button_frame, "Reset", self.reset, 2)
        self.exit_btn = self.create_button(self.button_frame, "Exit", root.quit, 3)

        self.progress_bar = ttk.Progressbar(root, orient="horizontal", length=500, mode="determinate")
        self.progress_bar.pack(pady=10)

        self.image_label = tk.Label(root, bg="#393e46", relief=tk.SUNKEN, width=60, height=25)
        self.image_label.pack(pady=10)

        self.result_label = tk.Label(root, text="No Image Processed", font=("Arial", 14), bg="#222831", fg="white")
        self.result_label.pack(pady=5)

        self.fake_real_label = tk.Label(root, text="", font=("Arial", 20, "bold"), bg="#222831")
        self.fake_real_label.pack(pady=5)

    def create_button(self, frame, text, command, col):
        return tk.Button(frame, text=text, command=command, font=("Arial", 12, "bold"), fg="white", bg="#00adb5", bd=3, relief=tk.RAISED, padx=10, pady=5).grid(row=0, column=col, padx=10, pady=5)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return
        self.progress_bar.start()
        self.executor.submit(self.process_image, file_path)

    def process_image(self, file_path):
        result = self.detector.predict(file_path)
        self.root.after(500, lambda: self.display_result(result))

    def display_result(self, result):
        self.progress_bar.stop()
        if result is None:
            self.result_label.config(text="Error processing image.")
            return
        img = Image.fromarray(result['original_image'])
        img.thumbnail((450, 450))
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img
        label_text = f"{result['filename']}\nConfidence: {result['confidence']}%"
        self.result_label.config(text=label_text, fg="red" if result['is_fake'] else "green")
        self.fake_real_label.config(text="FAKE" if result['is_fake'] else "REAL", fg="red" if result['is_fake'] else "green")

    def reset(self):
        self.image_label.config(image="")
        self.result_label.config(text="No Image Processed", fg="white")
        self.fake_real_label.config(text="", fg="white")

    def view_history(self):
        if not os.path.exists(CONFIG['history_file']):
            messagebox.showinfo("History", "No history available.")
            return
        with open(CONFIG['history_file'], 'r') as file:
            history = file.read()
        top = tk.Toplevel(self.root)
        top.title("Detection History")
        text = tk.Text(top, wrap=tk.WORD, width=50, height=15)
        text.insert(tk.END, history)
        text.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = DeepfakeDetectorApp(root)
    root.mainloop()
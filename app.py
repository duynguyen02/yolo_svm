import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2

class App:
    def __init__(self, root):
        root.eval('tk::PlaceWindow . center')
        self.root = root
        self.root.title("YOLO_SVM")

        self.model = None

        self.create_widgets()


    def create_widgets(self):
        self.pt_label = tk.Label(self.root, text="Select YOLO model:")
        self.pt_label.grid(row=0, column=0, padx=10, pady=10)
        self.pt_file_entry = tk.Entry(self.root, width=40)
        self.pt_file_entry.grid(row=0, column=1, padx=10, pady=10)
        self.pt_file_button = tk.Button(self.root, text="Browse", command=self.browse_pt_file)
        self.pt_file_button.grid(row=0, column=2, padx=10, pady=10)
        
        self.submit_button = tk.Button(self.root, text="Load model", command=self.load_model)
        self.submit_button.grid(row=1, column=0, columnspan=3, pady=10)

        self.image_label = tk.Label(self.root, text="Select image:")
        self.image_label.grid(row=2, column=0, padx=10, pady=10)
        self.image_file_entry = tk.Entry(self.root, width=40)
        self.image_file_entry.grid(row=2, column=1, padx=10, pady=10)
        self.image_file_button = tk.Button(self.root, text="Browse", command=self.browse_image_file)
        self.image_file_button.grid(row=2, column=2, padx=10, pady=10)

        self.additional_button = tk.Button(self.root, text="Submit", command=self.submit)
        self.additional_button.grid(row=3, column=0, columnspan=3, pady=10)

        self.additional_label = tk.Label(self.root, text="There are no models loaded yet", fg="blue")
        self.additional_label.grid(row=4, column=0, columnspan=3, pady=10)
        
        self.image_canvas = tk.Canvas(self.root, width=400, height=400)
        self.image_canvas.grid(row=5, column=0, columnspan=3, pady=10)

    def browse_pt_file(self):
        pt_file_path = filedialog.askopenfilename(filetypes=[("PT files", "*.pt")])
        if pt_file_path:
            self.pt_file_entry.delete(0, tk.END)
            self.pt_file_entry.insert(0, pt_file_path)

    def browse_image_file(self):
        image_file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg")])
        if image_file_path:
            self.image_file_entry.delete(0, tk.END)
            self.image_file_entry.insert(0, image_file_path)
            self.display_image(image_file_path)

    def display_image(self, image_path):
        img = Image.open(image_path)
        img = img.resize((400, 400), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img)
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def load_model(self):
        pt_file = self.pt_file_entry.get()
        try:
            self.model = YOLO(pt_file)
            model_names = ', '.join([self.model.names[i] for i in self.model.names])
            self.additional_label.config(text=f"Loaded model: {model_names[:100]}...")
        except Exception as e:
            messagebox.showinfo("Notification", f"Invalid model. Error: {e}")

    def submit(self):
        if self.model is None:
            messagebox.showinfo("Notification", "Please load the model.")
            return
            
        if not self.image_file_entry.get():
            messagebox.showinfo("Notification", "Please select the image.")
            return
        
        frame = cv2.imread(self.image_file_entry.get())
        results = self.model(frame)
        
        for result in results:
            result.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

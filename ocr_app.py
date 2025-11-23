import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import pytesseract
import csv
from datetime import datetime

class OCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("4D OCR Learning System")
        self.root.geometry("800x600")
        
        self.csv_file = "4d_results_history.csv"
        self.draw_count = self.count_draws()
        
        # Title
        tk.Label(root, text="4D OCR Learning System", font=("Arial", 20, "bold")).pack(pady=10)
        
        # Draw count display
        self.count_label = tk.Label(root, text=f"Learning from {self.draw_count} draws", 
                                     font=("Arial", 14), fg="blue")
        self.count_label.pack(pady=5)
        
        # OCR Button
        tk.Button(root, text="📷 OCR Scan", font=("Arial", 16, "bold"), 
                 bg="#4CAF50", fg="white", command=self.ocr_scan, 
                 width=20, height=2).pack(pady=20)
        
        # Image display
        self.img_label = tk.Label(root)
        self.img_label.pack(pady=10)
        
        # Result text
        self.result_text = tk.Text(root, height=10, width=70, font=("Courier", 10))
        self.result_text.pack(pady=10)
        
    def count_draws(self):
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                return sum(1 for _ in csv.reader(f)) - 1
        except:
            return 0
    
    def ocr_scan(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if not file_path:
            return
        
        # Display image
        img = Image.open(file_path)
        img.thumbnail((400, 300))
        photo = ImageTk.PhotoImage(img)
        self.img_label.config(image=photo)
        self.img_label.image = photo
        
        # OCR
        text = pytesseract.image_to_string(Image.open(file_path))
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
        
        # Update count
        self.draw_count += 1
        self.count_label.config(text=f"Learning from {self.draw_count} draws")
        messagebox.showinfo("Success", f"OCR Complete! Now learning from {self.draw_count} draws")

if __name__ == "__main__":
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()

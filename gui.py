import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
import numpy as np
from datetime import datetime
import time
from utils import load_image, save_image



from filters import (
    mean_filter,
    gaussian_filter,
    median_filter,
    adaptive_median_filter,
    bilateral_filter,
    apply_filter_to_channels,
    min_filter,
    max_filter,
)

FILTRI = {
    "Filtro di Media": {"func": mean_filter, "params": ["kernel_size"]},
    "Filtro Gaussiano": {"func": gaussian_filter, "params": ["kernel_size", "sigma"]},
    "Filtro Mediano": {"func": median_filter, "params": ["kernel_size"]},
    "Filtro Mediano Adattivo": {"func": adaptive_median_filter, "params": ["max_kernel_size"]},
    "Filtro Bilaterale": {"func": bilateral_filter, "params": ["kernel_size", "sigma_spatial", "sigma_intensity"]},
    "Filtro Minimo": {"func": min_filter, "params": ["kernel_size"]},
    "Filtro Massimo": {"func": max_filter, "params": ["kernel_size"]},
}


selected_image = None
output_folder = None
img_preview_before = None
img_preview_after = None
filtro_frames = []

class FiltroFrame(tk.Frame):
    def __init__(self, master, remove_callback, move_up_callback, move_down_callback):
        super().__init__(master)
        self.remove_callback = remove_callback
        self.move_up_callback = move_up_callback
        self.move_down_callback = move_down_callback

        self.filtro_var = tk.StringVar()
        self.filtro_var.set(list(FILTRI.keys())[0])
        self.filtro_menu = tk.OptionMenu(self, self.filtro_var, *FILTRI.keys(), command=self.update_parametri)
        self.filtro_menu.pack(side="left", padx=5, pady=5)

        self.parametri_frame = tk.Frame(self)
        self.parametri_frame.pack(side="left", padx=5, pady=5)
        self.parametri_entries = {}
        self.update_parametri(self.filtro_var.get())

        self.btn_up = tk.Button(self, text="↑", command=self.move_up_callback)
        self.btn_up.pack(side="left", padx=2)

        self.btn_down = tk.Button(self, text="↓", command=self.move_down_callback)
        self.btn_down.pack(side="left", padx=2)

        self.btn_remove = tk.Button(self, text="✖", command=self.remove_callback)
        self.btn_remove.pack(side="left", padx=2)

    def update_parametri(self, filtro_name):
        for widget in self.parametri_frame.winfo_children():
            widget.destroy()
        self.parametri_entries.clear()
        for param in FILTRI[filtro_name]["params"]:
            lbl = tk.Label(self.parametri_frame, text=param)
            lbl.pack(side="left")
            entry = tk.Entry(self.parametri_frame, width=5)
            entry.insert(0, "3")
            entry.pack(side="left", padx=2)
            self.parametri_entries[param] = entry

    def get_filtro(self):
        filtro_name = self.filtro_var.get()
        func = FILTRI[filtro_name]["func"]
        params = {}
        for param, entry in self.parametri_entries.items():
            val = entry.get()
            params[param] = float(val) if "." in val else int(val)
        return (filtro_name, func, params)

def scegli_immagine():
    global selected_image, img_preview_before
    path = filedialog.askopenfilename(filetypes=[("Immagini", "*.jpg *.png *.jpeg")])
    if path:
        selected_image = load_image(path)
        image = Image.open(path).resize((350, 350))
        img_preview_before = ImageTk.PhotoImage(image)
        canvas_before.create_image(0, 0, anchor="nw", image=img_preview_before)
        label_path.config(text=os.path.basename(path))

def scegli_cartella():
    global output_folder
    output_folder = filedialog.askdirectory()
    if output_folder:
        label_output.config(text=output_folder)

def aggiungi_filtro():
    def remove():
        filtro_frame.destroy()
        filtro_frames.remove(filtro_frame)

    def move_up():
        idx = filtro_frames.index(filtro_frame)
        if idx > 0:
            filtro_frames[idx], filtro_frames[idx - 1] = filtro_frames[idx - 1], filtro_frames[idx]
            refresh_filtro_list()

    def move_down():
        idx = filtro_frames.index(filtro_frame)
        if idx < len(filtro_frames) - 1:
            filtro_frames[idx], filtro_frames[idx + 1] = filtro_frames[idx + 1], filtro_frames[idx]
            refresh_filtro_list()

    filtro_frame = FiltroFrame(frame_filtri_lista, remove, move_up, move_down)
    filtro_frames.append(filtro_frame)
    refresh_filtro_list()

def refresh_filtro_list():
    for widget in frame_filtri_lista.winfo_children():
        widget.pack_forget()
    for frame in filtro_frames:
        frame.pack(fill="x", pady=3)

def stats(img):
    if img.ndim == 2:
        return [(np.mean(img), np.var(img))]
    return [(np.mean(img[:, :, c]), np.var(img[:, :, c])) for c in range(img.shape[2])]

def applica_filtri():
    global selected_image, output_folder, img_preview_after
    if selected_image is None:
        messagebox.showerror("Errore", "Carica prima un'immagine.")
        return
    if output_folder is None:
        messagebox.showerror("Errore", "Seleziona una cartella di output.")
        return
    if not filtro_frames:
        messagebox.showerror("Errore", "Aggiungi almeno un filtro.")
        return

    os.makedirs(output_folder, exist_ok=True)

    filtri_da_applicare = [f.get_filtro() for f in filtro_frames]

    filtered = selected_image
    start_time = time.time()
    descrizioni = []

    progress["maximum"] = len(filtri_da_applicare)
    for i, (nome, func, params) in enumerate(filtri_da_applicare):
        step_start = time.time()
        filtered = apply_filter_to_channels(filtered, func, **params)
        step_time = time.time() - step_start

        descrizioni.append(f"{nome} {params}")
        progress["value"] = i + 1

        pct = int((i + 1) / len(filtri_da_applicare) * 100)
        percent.set(f"{pct}%")

        remaining = (len(filtri_da_applicare) - (i + 1)) * step_time
        time_remaining_text.set(f"Tempo stimato rimanente: {remaining:.1f} sec")

        root.update_idletasks()

    end_time = time.time()

    output_path = os.path.join(output_folder, "output.jpg")
    log_path = os.path.join(output_folder, "log.txt")
    save_image(filtered, output_path)

    stats_before = stats(selected_image)
    stats_after = stats(filtered)

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("LOG ELABORAZIONE IMMAGINE\n\n")
        for i, (b, a) in enumerate(zip(stats_before, stats_after)):
            delta = b[1] - a[1]
            perc_var = (delta / b[1]) * 100 if b[1] != 0 else 0
            log.write(f"Canale {i}: prima μ={b[0]:.2f}, σ²={b[1]:.2f} → dopo μ={a[0]:.2f}, σ²={a[1]:.2f} ➜ -{perc_var:.1f}%\n")
        log.write("\nFiltri applicati:\n")
        for d in descrizioni:
            log.write(f" - {d}\n")
        log.write(f"\nTempo esecuzione: {end_time - start_time:.2f} sec\n")
        log.write(f"Salvato in: {output_path}\n")
        log.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    image_out = Image.fromarray(filtered.astype(np.uint8)).resize((350, 350))
    img_preview_after = ImageTk.PhotoImage(image_out)
    canvas_after.create_image(0, 0, anchor="nw", image=img_preview_after)

    messagebox.showinfo("✅ Fatto", f"Filtri applicati correttamente.\nFile salvati in:\n{output_folder}")
    # Reset UI per nuova esecuzione
    progress["value"] = 0
    percent.set("0%")
    time_remaining_text.set("Tempo stimato rimanente: --")


# === INTERFACCIA GRAFICA ===

root = tk.Tk()
root.title("Image Denoising - GUI")
root.geometry("1100x700")
percent = tk.StringVar(value="0%")
time_remaining_text = tk.StringVar(value="Tempo stimato rimanente: --")

# Frame sinistro
frame_img = tk.Frame(root)
frame_img.pack(side="left", padx=10, pady=10)

tk.Label(frame_img, text="PRIMA").pack()
canvas_before = tk.Canvas(frame_img, width=350, height=350, bg="lightgray")
canvas_before.pack()

tk.Label(frame_img, text="DOPO").pack()
canvas_after = tk.Canvas(frame_img, width=350, height=350, bg="lightgray")
canvas_after.pack()

# Frame destro
frame_ctrl = tk.Frame(root)
frame_ctrl.pack(side="right", fill="both", expand=True, padx=10, pady=10)

tk.Button(frame_ctrl, text="📂 Carica immagine", command=scegli_immagine).pack()
label_path = tk.Label(frame_ctrl, text="Nessuna immagine selezionata")
label_path.pack(pady=5)

tk.Button(frame_ctrl, text="📁 Seleziona cartella output", command=scegli_cartella).pack()
label_output = tk.Label(frame_ctrl, text="Nessuna cartella selezionata")
label_output.pack(pady=5)

tk.Label(frame_ctrl, text="🎛️ Filtri da applicare").pack(pady=(10, 0))
frame_filtri_lista = tk.Frame(frame_ctrl)
frame_filtri_lista.pack(pady=5, fill="x")

tk.Button(frame_ctrl, text="➕ Aggiungi filtro", command=aggiungi_filtro).pack(pady=5)

progress = ttk.Progressbar(frame_ctrl, length=300, mode="determinate")
progress.pack(pady=(10, 0))
tk.Label(frame_ctrl, textvariable=percent).pack()
tk.Label(frame_ctrl, textvariable=time_remaining_text).pack()

tk.Button(frame_ctrl, text="🧪 Applica filtri", command=applica_filtri, bg="green", fg="white").pack(pady=20)

root.mainloop()

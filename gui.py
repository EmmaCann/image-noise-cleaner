import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import numpy as np
from datetime import datetime
import time

from filters import (
    mean_filter,
    gaussian_filter,
    median_filter,
    adaptive_median_filter,
    bilateral_filter,
    apply_filter_to_channels,
)
from utils import save_image, load_image

# Filtri disponibili
FILTRI = {
    "Filtro di Media": {"func": mean_filter, "params": ["kernel_size"]},
    "Filtro Gaussiano": {"func": gaussian_filter, "params": ["kernel_size", "sigma"]},
    "Filtro Mediano": {"func": median_filter, "params": ["kernel_size"]},
    "Filtro Mediano Adattivo": {"func": adaptive_median_filter, "params": ["max_kernel_size"]},
    "Filtro Bilaterale": {"func": bilateral_filter, "params": ["kernel_size", "sigma_spatial", "sigma_intensity"]},
}

# Variabili globali
selected_image = None
output_folder = None

def scegli_immagine():
    global selected_image, img_preview
    path = filedialog.askopenfilename(filetypes=[("Immagini", "*.jpg *.png *.jpeg")])
    if path:
        selected_image = load_image(path)
        image = Image.open(path).resize((250, 250))
        img_preview = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor="nw", image=img_preview)
        label_path.config(text=f"Immagine: {os.path.basename(path)}")

def scegli_cartella():
    global output_folder
    output_folder = filedialog.askdirectory()
    if output_folder:
        label_output.config(text=f"Cartella: {output_folder}")

def applica_filtro():
    global selected_image, output_folder
    if selected_image is None:
        messagebox.showerror("Errore", "Seleziona prima un'immagine.")
        return
    if output_folder is None:
        messagebox.showerror("Errore", "Seleziona una cartella di salvataggio.")
        return

    nome_filtro = filtro_var.get()
    filtro_info = FILTRI.get(nome_filtro)
    if not filtro_info:
        messagebox.showerror("Errore", "Filtro non valido.")
        return

    try:
        params = {}
        for p in filtro_info["params"]:
            val = float(entry_params[p].get()) if "." in entry_params[p].get() else int(entry_params[p].get())
            params[p] = val
    except ValueError:
        messagebox.showerror("Errore", "Inserisci parametri validi.")
        return

    os.makedirs(output_folder, exist_ok=True)

    # Applica filtro
    start_time = time.time()
    filtered = apply_filter_to_channels(selected_image, filtro_info["func"], **params)
    end_time = time.time()

    output_path = os.path.join(output_folder, "output.jpg")
    log_path = os.path.join(output_folder, "log.txt")
    save_image(filtered, output_path)

    # Log
    def stats(img):
        if img.ndim == 2:
            return [(np.mean(img), np.var(img))]
        return [(np.mean(img[:, :, c]), np.var(img[:, :, c])) for c in range(img.shape[2])]

    stats_before = stats(selected_image)
    stats_after = stats(filtered)

    with open(log_path, "w", encoding="utf-8") as log:
        log.write("LOG ELABORAZIONE IMMAGINE\n\n")
        log.write(f"Filtro: {nome_filtro}\n")
        log.write(f"Parametri: {params}\n\n")
        log.write("STATISTICHE (media, varianza):\n")
        for i, (b, a) in enumerate(zip(stats_before, stats_after)):
            delta = b[1] - a[1]
            perc = (delta / b[1]) * 100 if b[1] != 0 else 0
            log.write(f"Canale {i}: prima (μ={b[0]:.2f}, σ²={b[1]:.2f}), dopo (μ={a[0]:.2f}, σ²={a[1]:.2f}) ➜ var ridotta del {perc:.1f}%\n")
        log.write(f"\n🕒 Tempo elaborazione: {end_time - start_time:.2f} secondi\n")
        log.write(f"📁 Immagine salvata in: {output_path}\n")
        log.write(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    messagebox.showinfo("Successo", f"Filtro applicato.\nFile salvati in:\n{output_folder}")

# UI Setup
root = tk.Tk()
root.title("Image Denoising GUI")
root.geometry("500x600")

btn_load = tk.Button(root, text="📂 Carica immagine", command=scegli_immagine)
btn_load.pack(pady=5)

canvas = tk.Canvas(root, width=250, height=250, bg="gray")
canvas.pack()

label_path = tk.Label(root, text="Immagine: nessuna")
label_path.pack()

tk.Label(root, text="Scegli un filtro:").pack()
filtro_var = tk.StringVar(value=list(FILTRI.keys())[0])
menu = tk.OptionMenu(root, filtro_var, *FILTRI.keys())
menu.pack()

entry_params = {}
frame_params = tk.Frame(root)
frame_params.pack(pady=5)

def aggiorna_parametri(*args):
    for widget in frame_params.winfo_children():
        widget.destroy()
    filtro_info = FILTRI.get(filtro_var.get())
    if filtro_info:
        for p in filtro_info["params"]:
            tk.Label(frame_params, text=p).pack()
            entry = tk.Entry(frame_params)
            entry.pack()
            entry.insert(0, "3")  # valore di default
            entry_params[p] = entry

filtro_var.trace_add("write", aggiorna_parametri)
aggiorna_parametri()

btn_cartella = tk.Button(root, text="📁 Seleziona cartella di salvataggio", command=scegli_cartella)
btn_cartella.pack(pady=5)

label_output = tk.Label(root, text="Cartella: nessuna")
label_output.pack()

btn_applica = tk.Button(root, text="🧪 Applica filtro", command=applica_filtro, bg="green", fg="white")
btn_applica.pack(pady=20)

root.mainloop()

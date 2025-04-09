import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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

# === CONFIGURAZIONE FILTRI ===
FILTRI = {
    "Filtro di Media": {"func": mean_filter, "params": ["kernel_size"]},
    "Filtro Gaussiano": {"func": gaussian_filter, "params": ["kernel_size", "sigma"]},
    "Filtro Mediano": {"func": median_filter, "params": ["kernel_size"]},
    "Filtro Mediano Adattivo": {"func": adaptive_median_filter, "params": ["max_kernel_size"]},
    "Filtro Bilaterale": {"func": bilateral_filter, "params": ["kernel_size", "sigma_spatial", "sigma_intensity"]},
}

selected_image = None
output_folder = None
img_preview_before = None
img_preview_after = None

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

def aggiorna_parametri(*args):
    for widget in frame_params.winfo_children():
        widget.destroy()
    entry_params.clear()
    selezionati = listbox.curselection()
    for i in selezionati:
        nome = listbox.get(i)
        for p in FILTRI[nome]["params"]:
            frame = tk.Frame(frame_params)
            tk.Label(frame, text=f"{nome} - {p}").pack(side="left")
            entry = tk.Entry(frame, width=6)
            entry.insert(0, "3")
            entry.pack(side="right")
            frame.pack(pady=1)
            entry_params[f"{nome}:{p}"] = entry

def stats(img):
    if img.ndim == 2:
        return [(np.mean(img), np.var(img))]
    return [(np.mean(img[:, :, c]), np.var(img[:, :, c])) for c in range(img.shape[2])]

def toggle_filtri():
    if frame_filtri.winfo_viewable():
        frame_filtri.pack_forget()
        btn_toggle_filtri.config(text="➕ Mostra filtri")
    else:
        frame_filtri.pack(pady=5, fill="x")
        btn_toggle_filtri.config(text="➖ Nascondi filtri")

def applica_filtri():
    global selected_image, output_folder, img_preview_after
    if selected_image is None:
        messagebox.showerror("Errore", "Carica prima un'immagine.")
        return
    if output_folder is None:
        messagebox.showerror("Errore", "Seleziona una cartella di output.")
        return
    selezionati = listbox.curselection()
    if not selezionati:
        messagebox.showerror("Errore", "Seleziona almeno un filtro.")
        return

    os.makedirs(output_folder, exist_ok=True)

    filtri_da_applicare = []
    for i in selezionati:
        nome = listbox.get(i)
        func = FILTRI[nome]["func"]
        params = {}
        for p in FILTRI[nome]["params"]:
            val = entry_params[f"{nome}:{p}"].get()
            params[p] = float(val) if "." in val else int(val)
        filtri_da_applicare.append((nome, func, params))

    filtered = selected_image
    start_time = time.time()
    descrizioni = []

    progress["maximum"] = len(filtri_da_applicare)
    for i, (nome, func, params) in enumerate(filtri_da_applicare):
        filtered = apply_filter_to_channels(filtered, func, **params)
        descrizioni.append(f"{nome} {params}")
        progress["value"] = i + 1
        percent.set(f"{int((i + 1) / len(filtri_da_applicare) * 100)}%")
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

    # Mostra immagine filtrata
    image_out = Image.fromarray(filtered.astype(np.uint8)).resize((350, 350))
    img_preview_after = ImageTk.PhotoImage(image_out)
    canvas_after.create_image(0, 0, anchor="nw", image=img_preview_after)

    messagebox.showinfo("✅ Fatto", f"Filtri applicati correttamente.\nFile salvati in:\n{output_folder}")

# === INTERFACCIA GRAFICA ===

root = tk.Tk()
root.title("Image Denoising - GUI")
root.geometry("1000x700")
entry_params = {}
percent = tk.StringVar(value="0%")

# Frame sinistro (immagini)
frame_img = tk.Frame(root)
frame_img.pack(side="left", padx=10, pady=10)

tk.Label(frame_img, text="PRIMA").pack()
canvas_before = tk.Canvas(frame_img, width=350, height=350, bg="lightgray")
canvas_before.pack()

tk.Label(frame_img, text="DOPO").pack()
canvas_after = tk.Canvas(frame_img, width=350, height=350, bg="lightgray")
canvas_after.pack()

# Frame destro (controlli)
frame_ctrl = tk.Frame(root)
frame_ctrl.pack(side="right", fill="both", expand=True, padx=10, pady=10)

tk.Button(frame_ctrl, text="📂 Carica immagine", command=scegli_immagine).pack()
label_path = tk.Label(frame_ctrl, text="Nessuna immagine selezionata")
label_path.pack(pady=5)

tk.Button(frame_ctrl, text="📁 Seleziona cartella output", command=scegli_cartella).pack()
label_output = tk.Label(frame_ctrl, text="Nessuna cartella selezionata")
label_output.pack(pady=5)

btn_toggle_filtri = tk.Button(frame_ctrl, text="➕ Mostra filtri", command=toggle_filtri)
btn_toggle_filtri.pack(pady=(10, 0))

# Contenitore filtri e parametri (a comparsa)
frame_filtri = tk.Frame(frame_ctrl)
listbox = tk.Listbox(frame_filtri, selectmode=tk.MULTIPLE, height=6)
for nome in FILTRI:
    listbox.insert(tk.END, nome)
listbox.pack(pady=5, fill="x")
listbox.bind("<<ListboxSelect>>", aggiorna_parametri)

frame_params = tk.Frame(frame_filtri)
frame_params.pack(pady=5)
entry_params = {}

# Barra di avanzamento
progress = ttk.Progressbar(frame_ctrl, length=300, mode="determinate")
progress.pack(pady=(10, 0))
label_percent = tk.Label(frame_ctrl, textvariable=percent)
label_percent.pack()

# Esegui
tk.Button(frame_ctrl, text="🧪 Applica filtri", command=applica_filtri, bg="green", fg="white").pack(pady=20)

root.mainloop()

from utils import load_image, save_image, create_output_folder, add_noise
from filters import mean_filter, gaussian_filter, median_filter, adaptive_median_filter, bilateral_filter
from filters import apply_filter_to_channels
import numpy as np
import time
from datetime import datetime

# Dizionario dei filtri disponibili
filtro_catalogo = {
    "1": {"name": "Filtro di Media", "func": mean_filter, "params": ["kernel_size"]},
    "2": {"name": "Filtro Gaussiano", "func": gaussian_filter, "params": ["kernel_size", "sigma"]},
    "3": {"name": "Filtro Mediano", "func": median_filter, "params": ["kernel_size"]},
    "4": {"name": "Filtro Mediano Adattivo", "func": adaptive_median_filter, "params": ["max_kernel_size"]},
    "5": {"name": "Filtro Bilaterale", "func": bilateral_filter, "params": ["kernel_size", "sigma_spatial", "sigma_intensity"]},
}

def scegli_filtri():
    print("\n Filtri disponibili:")
    for k, v in filtro_catalogo.items():
        print(f"{k}. {v['name']}")

    selezione = input("\nInserisci i numeri dei filtri da applicare (es: 1 3 5): ").split()
    sequenza = []

    for scelta in selezione:
        if scelta in filtro_catalogo:
            filtro = filtro_catalogo[scelta]
            parametri = {}
            print(f"\n  Parametri per {filtro['name']}:")
            for p in filtro["params"]:
                val = input(f" - {p}: ")
                parametri[p] = float(val) if "." in val else int(val)
            sequenza.append({
                "name": filtro["name"],
                "func": filtro["func"],
                "params": parametri
            })
        else:
            print(f"⚠️  Filtro {scelta} non valido, ignorato.")
    return sequenza

def calc_stats(img):
    if img.ndim == 2:
        return [np.mean(img), np.var(img)]
    else:
        return [(np.mean(img[:, :, c]), np.var(img[:, :, c])) for c in range(img.shape[2])]

def main():
    input_path = "images/balloons_noisy.png"
    # noisy_path = "output/noisy.jpg"
    output_image_path = "output/output.jpg"
    output_log_path = "output/log.txt"

    create_output_folder()

    # Carica immagine originale
    image = load_image(input_path)

    # Rumore artificiale 
    # noisy_image = add_noise(image, noise_type="salt_pepper", amount=15)
    # save_image(noisy_image, noisy_path)
    # image = noisy_image

    # Statistiche pre-filtraggio
    stats_before = calc_stats(image)

    # Richiesta filtri all’utente
    filters_to_apply = scegli_filtri()

    # Applica i filtri scelti
    filtered_image = image
    descrizioni = []
    start_time = time.time()

    for filtro in filters_to_apply:
        filtered_image = apply_filter_to_channels(filtered_image, filtro["func"], **filtro["params"])
        descrizioni.append(f"{filtro['name']} {filtro['params']}")

    execution_time = time.time() - start_time
    stats_after = calc_stats(filtered_image)

    # Salva immagine finale
    save_image(filtered_image, output_image_path)

    # Log dettagliato
    with open(output_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("========== LOG ELABORAZIONE IMMAGINE ==========\n\n")
        log_file.write(f" File: {input_path}\n")
        log_file.write(f" Dimensioni: {image.shape[0]} x {image.shape[1]} x {image.shape[2] if image.ndim == 3 else 1}\n")
        log_file.write(f" Tipo immagine: {'RGB' if image.ndim == 3 else 'Grayscale'}\n\n")

        # log_file.write("Rumore artificiale: Salt & Pepper (amount = 15%)\n\n")

        log_file.write(" FILTRI APPLICATI:\n")
        for i, desc in enumerate(descrizioni, 1):
            log_file.write(f"  {i}. {desc}\n")

        log_file.write("\n STATISTICHE\n")

        def format_stats(label, stats):
            if isinstance(stats[0], tuple):  # RGB
                for idx, (mean, var) in enumerate(stats):
                    log_file.write(f"{label} - Canale {idx}: media={mean:.2f}, varianza={var:.2f}\n")
            else:
                mean, var = stats
                log_file.write(f"{label}: media={mean:.2f}, varianza={var:.2f}\n")

        format_stats("PRIMA", stats_before)
        format_stats("DOPO", stats_after)

        if isinstance(stats_before[0], tuple):
            log_file.write("\n RIDUZIONE VARIANZA per canale:\n")
            for i in range(len(stats_before)):
                v_before = stats_before[i][1]
                v_after = stats_after[i][1]
                perc = 100 * (v_before - v_after) / v_before if v_before != 0 else 0
                log_file.write(f"  Canale {i}: -{perc:.2f}%\n")

        log_file.write(f"\n Tempo di esecuzione: {execution_time:.2f} secondi\n")
        log_file.write(f" Immagine salvata in: {output_image_path}\n")
        log_file.write(f" Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        log_file.write("\n================================================\n")

    print("\n Filtri applicati con successo. Immagini e log salvati.")

if __name__ == "__main__":
    main()

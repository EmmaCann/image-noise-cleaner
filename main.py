from utils import load_image, save_image, create_output_folder, add_noise
from filters import mean_filter, gaussian_filter, median_filter, adaptive_median_filter, bilateral_filter
import numpy as np

# Dizionario dei filtri disponibili
filtro_catalogo = {
    "1": {"name": "Filtro di Media", "func": mean_filter, "params": ["kernel_size"]},
    "2": {"name": "Filtro Gaussiano", "func": gaussian_filter, "params": ["kernel_size", "sigma"]},
    "3": {"name": "Filtro Mediano", "func": median_filter, "params": ["kernel_size"]},
    "4": {"name": "Filtro Mediano Adattivo", "func": adaptive_median_filter, "params": ["max_kernel_size"]},
    "5": {"name": "Filtro Bilaterale", "func": bilateral_filter, "params": ["kernel_size", "sigma_spatial", "sigma_intensity"]},
}

def scegli_filtri():
    print("\n🎛️  Filtri disponibili:")
    for k, v in filtro_catalogo.items():
        print(f"{k}. {v['name']}")

    selezione = input("\nInserisci i numeri dei filtri da applicare (es: 1 3 5): ").split()
    sequenza = []

    for scelta in selezione:
        if scelta in filtro_catalogo:
            filtro = filtro_catalogo[scelta]
            parametri = {}
            print(f"\n➡️  Parametri per {filtro['name']}:")
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

def main():
    input_path = "images/balloons_noisy.png"
    ##noisy_path = "output/noisy.jpg"
    output_image_path = "output/output.jpg"
    output_log_path = "output/log.txt"

    create_output_folder()

    # Carica immagine originale
    image = load_image(input_path)

    # Aggiungi rumore artificiale
   ## noisy_image = add_noise(image, noise_type="salt_pepper", amount=15)
    ##save_image(noisy_image, noisy_path)

    # Richiesta filtri all’utente
    filters_to_apply = scegli_filtri()

    # Applica i filtri scelti
    #filtered_image = noisy_image
    filtered_image = image
    descrizioni = []

    for filtro in filters_to_apply:
        filtered_image = filtro["func"](filtered_image, **filtro["params"])
        descrizioni.append(f"{filtro['name']} {filtro['params']}")

    # Salva immagine finale
    save_image(filtered_image, output_image_path)

    # Statistiche
    #mean_before = np.mean(noisy_image)
    mean_after = np.mean(filtered_image)
    #var_before = np.var(noisy_image)
    var_after = np.var(filtered_image)

    # Log dettagliato
    with open(output_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("LOG DEL PROGRAMMA\n")
        log_file.write("Rumore aggiunto: sale e pepe (amount = 30)\n\n")
        log_file.write("Filtri applicati in sequenza:\n")
        for desc in descrizioni:
            log_file.write(f"  - {desc}\n")
        log_file.write("\n")
        #log_file.write(f"Media pixel prima dei filtri: {mean_before:.2f}\n")
        log_file.write(f"Media pixel dopo i filtri: {mean_after:.2f}\n")
        #log_file.write(f"Varianza prima dei filtri: {var_before:.2f}\n")
        log_file.write(f"Varianza dopo i filtri: {var_after:.2f}\n")
       # log_file.write(f"\nImmagini salvate in:\n - {noisy_path} (rumore)\n - {output_image_path} (filtrata)\n")

    print("\n✅ Filtri applicati con successo. Immagini e log salvati.")

if __name__ == "__main__":
    main()

from utils import load_image, save_image, create_output_folder, add_noise
from filters import mean_filter
import numpy as np

def main():
    input_path = "images/input.jpg"
    noisy_path = "output/noisy.jpg"
    output_image_path = "output/output.jpg"
    output_log_path = "output/log.txt"

    create_output_folder()

    # 1. Carica immagine originale
    image = load_image(input_path)

    # 2. Aggiungi rumore artificiale (es. gaussiano)
    noisy_image = add_noise(image, noise_type="gaussian", amount=30)
    save_image(noisy_image, noisy_path)

    # 3. Applica filtro di media
    filtered_image = mean_filter(noisy_image, kernel_size=3)
    save_image(filtered_image, output_image_path)

    # 4. Dati statistici per il log
    mean_before = np.mean(noisy_image)
    mean_after = np.mean(filtered_image)
    var_before = np.var(noisy_image)
    var_after = np.var(filtered_image)

    # 5. Log dettagliato
    with open(output_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("LOG DEL PROGRAMMA\n")
        log_file.write("Filtro applicato: Filtro di Media (3x3)\n")
        log_file.write("Rumore aggiunto: Gaussiano (σ = 30)\n")
        log_file.write(f"Media pixel prima del filtro: {mean_before:.2f}\n")
        log_file.write(f"Media pixel dopo il filtro: {mean_after:.2f}\n")
        log_file.write(f"Varianza prima del filtro: {var_before:.2f}\n")
        log_file.write(f"Varianza dopo il filtro: {var_after:.2f}\n")
        log_file.write(f"Immagini salvate in: {output_image_path}, {noisy_path}\n")

    print("Filtro applicato con successo. Log e immagini salvati.")

if __name__ == "__main__":
    main()

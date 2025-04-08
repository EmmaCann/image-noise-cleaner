from utils import load_image, save_image, create_output_folder, add_noise
from filters import mean_filter, gaussian_filter,median_filter,adaptive_median_filter,bilateral_filter,non_local_means_filter
import numpy as np

def main():
    input_path = "images/input.png"
    noisy_path = "output/noisy.jpg"
    output_image_path = "output/output.jpg"
    output_log_path = "output/log.txt"

    create_output_folder()

    # Carica immagine originale
    image = load_image(input_path)
    
    #  Aggiungi rumore artificiale (es. gaussiano)
    noisy_image = add_noise(image, noise_type="gaussian", amount=30)
    noisy_image = noisy_image[:100, :100]

    save_image(noisy_image, noisy_path)

  

    # Filtro di media
    # filtered_image = mean_filter(noisy_image, kernel_size=3)
    # filtro_descrizione = "Filtro di Media (3x3)"

    # Filtro Gaussiano
    #filtered_image = gaussian_filter(image, kernel_size=5, sigma=1.0)
    #filtro_descrizione = "Filtro Gaussiano (5x5, σ = 1.0)"

    # Filtro Mediano
    #filtered_image = median_filter(noisy_image, kernel_size=3)
    #filtro_descrizione = "Filtro Mediano (3x3)"

    # Filtro Mediano Adattivo
    #filtered_image = adaptive_median_filter(image, max_kernel_size=7)
    #filtro_descrizione = "Filtro Mediano Adattivo (fino a 7x7)"

    # Filtro Bilaterale
    filtered_image = bilateral_filter(image, kernel_size=5, sigma_spatial=2.0, sigma_intensity=30.0)
    filtro_descrizione = "Filtro Bilaterale (5x5, σ_spaziale = 2.0, σ_intensità = 30.0)"

    save_image(filtered_image, output_image_path)

    #  Dati statistici per il log
    mean_before = np.mean(image)
    mean_after = np.mean(filtered_image)
    var_before = np.var(image)
    var_after = np.var(filtered_image)

    #  Log dettagliato
    with open(output_log_path, "w", encoding="utf-8") as log_file:
        log_file.write("LOG DEL PROGRAMMA\n")
        log_file.write(f"Filtro applicato: {filtro_descrizione}\n")
        log_file.write("Rumore aggiunto: Gaussiano (σ = 30)\n")
        log_file.write(f"Media pixel prima del filtro: {mean_before:.2f}\n")
        log_file.write(f"Media pixel dopo il filtro: {mean_after:.2f}\n")
        log_file.write(f"Varianza prima del filtro: {var_before:.2f}\n")
        log_file.write(f"Varianza dopo il filtro: {var_after:.2f}\n")
        log_file.write(f"Immagini salvate in: {output_image_path}, {noisy_path}\n")

    print("Filtro applicato con successo. Log e immagini salvati.")

if __name__ == "__main__":
    main()

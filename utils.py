from PIL import Image
import numpy as np
import os

def load_image(path):
    """
    Carica un'immagine dal percorso specificato e la converte in scala di grigi.
    Ritorna l'immagine come array NumPy.
    """
    image = Image.open(path).convert("L")  # "L" = modalità grayscale
    return np.array(image)

def save_image(image_array, path):
    """
    Salva un array NumPy come immagine JPEG/PNG.
    """
    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(path)

def create_output_folder(folder_name="output"):
    """
    Crea la cartella di output se non esiste già.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)



def add_noise(image, noise_type="gaussian", amount=30):
    """
    Aggiunge rumore all'immagine.
    - noise_type: 'gaussian' o 'salt_pepper'
    - amount: intensità del rumore (più alto = più rumore)
    Ritorna una nuova immagine con rumore.
    """
    noisy = image.copy()

    if noise_type == "gaussian":
        gaussian_noise = np.random.normal(0, amount, image.shape)
        noisy = image + gaussian_noise
        noisy = np.clip(noisy, 0, 255)

    elif noise_type == "salt_pepper":
        prob = amount / 100  # es. 0.05 = 5% dei pixel
        noisy = image.copy()
        rnd = np.random.rand(*image.shape)
        noisy[rnd < prob / 2] = 0
        noisy[rnd > 1 - prob / 2] = 255

    return noisy.astype(np.uint8)

from PIL import Image
import numpy as np
import os

def load_image(path):
    """
    Carica un'immagine dal percorso specificato e la restituisce come array NumPy.
    Supporta RGB.
    """
    image = Image.open(path).convert("RGB")  # "L" = modalità grayscale
    return np.array(image)

def save_image(image_array, path):
    """
    Salva un array NumPy come immagine JPEG/PNG.
    Supporta sia RGB che scala di grigi.
    """
    image_array = image_array.astype(np.uint8)
    if image_array.ndim == 3 and image_array.shape[2] == 3:
        image = Image.fromarray(image_array, mode='RGB')
    else:
        image = Image.fromarray(image_array)
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
    Supporta RGB.
    """
    noisy = image.copy()

    if noise_type == "gaussian":
        gaussian_noise = np.random.normal(0, amount, image.shape)
        noisy = image + gaussian_noise
        noisy = np.clip(noisy, 0, 255)

    elif noise_type == "salt_pepper":
        prob = amount / 100
        noisy = image.copy()
        if image.ndim == 3:
            for c in range(3):
                rnd = np.random.rand(*image[:, :, c].shape)
                noisy[:, :, c][rnd < prob / 2] = 0
                noisy[:, :, c][rnd > 1 - prob / 2] = 255
        else:
            rnd = np.random.rand(*image.shape)
            noisy[rnd < prob / 2] = 0
            noisy[rnd > 1 - prob / 2] = 255

    return noisy.astype(np.uint8)

import numpy as np

def mean_filter(image, kernel_size=3):
    """
    Applica un filtro di media a un'immagine (array NumPy in scala di grigi).
    kernel_size: dimensione del filtro (deve essere dispari, es: 3, 5, 7...)
    """
    padded_image = np.pad(image, pad_width=kernel_size//2, mode='edge')
    filtered_image = np.zeros_like(image)

    height, width = image.shape

    for i in range(height):
        for j in range(width):
            # Estraggo finestra (patch) centrata su (i, j)
            window = padded_image[i:i+kernel_size, j:j+kernel_size]
            # Calcolo la media della finestra
            filtered_image[i, j] = np.mean(window)

    return filtered_image

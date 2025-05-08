import numpy as np


def apply_filter_to_channels(image, filter_func, **kwargs):
    if image.ndim == 3:  # immagine RGB
        channels = []
        for c in range(3):
            channel_filtered = filter_func(image[:, :, c], **kwargs)
            channels.append(channel_filtered)
        return np.stack(channels, axis=2)
    else:
        return filter_func(image, **kwargs)

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



def gaussian_kernel(size, sigma=1):
    """
    Crea un kernel gaussiano 2D normalizzato.
    """
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)

def gaussian_filter(image, kernel_size=3, sigma=1):
    """
    Applica un filtro gaussiano a un'immagine.
    - image: array 2D grayscale
    - kernel_size: dimensione del filtro (deve essere dispari)
    - sigma: deviazione standard della gaussiana
    """
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')
    filtered_image = np.zeros_like(image)

    height, width = image.shape

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_pixel = np.sum(region * kernel)
            filtered_image[i, j] = filtered_pixel

    return filtered_image



def median_filter(image, kernel_size=3):
    """
    Applica un filtro mediano a un'immagine (grayscale).
    kernel_size: dimensione del filtro (es. 3, 5, 7...)
    """
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')
    filtered_image = np.zeros_like(image)

    height, width = image.shape

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            median_value = np.median(region)
            filtered_image[i, j] = median_value

    return filtered_image



def adaptive_median_filter(image, max_kernel_size=7):
    """
    Applica un filtro mediano adattivo.
    - max_kernel_size: dimensione massima della finestra (es. 7)
    """
    padded_image = np.pad(image, max_kernel_size // 2, mode='edge')
    filtered_image = np.zeros_like(image)
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            current_size = 3
            pixel_filtered = False

            while current_size <= max_kernel_size and not pixel_filtered:
                pad = current_size // 2
                region = padded_image[i:i+current_size, j:j+current_size]
                z_min = np.min(region)
                z_max = np.max(region)
                z_med = np.median(region)
                z_xy = padded_image[i + pad, j + pad]

                A1 = z_med - z_min
                A2 = z_med - z_max

                if A1 > 0 and A2 < 0:
                    B1 = z_xy - z_min
                    B2 = z_xy - z_max
                    if B1 > 0 and B2 < 0:
                        filtered_image[i, j] = z_xy  # pixel non rumoroso
                    else:
                        filtered_image[i, j] = z_med  # pixel rumoroso → sostituito
                    pixel_filtered = True
                else:
                    current_size += 2  # aumenta dimensione finestra

            if not pixel_filtered:
                filtered_image[i, j] = z_med  # fallback: usa il mediano finale

    return filtered_image


import numpy as np

def bilateral_filter(image, kernel_size=5, sigma_spatial=2.0, sigma_intensity=30.0):
    """
    Applica un filtro bilaterale fatto a mano.
    - kernel_size: dimensione della finestra
    - sigma_spatial: influenza della distanza (es. 2.0)
    - sigma_intensity: influenza della differenza di intensità (es. 30.0)
    """
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='edge')
    filtered_image = np.zeros_like(image, dtype=np.float64)

    # Pre-calcola i pesi spaziali (dipendono solo dalla posizione)
    spatial_weights = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            dx = i - pad
            dy = j - pad
            spatial_weights[i, j] = np.exp(-(dx**2 + dy**2) / (2 * sigma_spatial**2))

    height, width = image.shape

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            intensity_diff = region - image[i, j]
            radiometric_weights = np.exp(-(intensity_diff**2) / (2 * sigma_intensity**2))
            total_weights = spatial_weights * radiometric_weights
            weight_sum = np.sum(total_weights)
            pixel_value = np.sum(region * total_weights) / weight_sum
            filtered_image[i, j] = pixel_value

    return np.clip(filtered_image, 0, 255).astype(np.uint8)



def min_filter(image, kernel_size=3):
    """
    Filtro Minimo: sostituisce ogni pixel con il valore minimo nella finestra.
    Utile per rimuovere rumore impulsivo chiaro.
    """
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='edge')
    filtered_image = np.zeros_like(image)

    height, width = image.shape

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.min(region)

    return filtered_image

def max_filter(image, kernel_size=3):
    """
    Filtro Massimo: sostituisce ogni pixel con il valore massimo nella finestra.
    Utile per rimuovere rumore impulsivo scuro.
    """
    pad = kernel_size // 2
    padded_image = np.pad(image, pad, mode='edge')
    filtered_image = np.zeros_like(image)

    height, width = image.shape

    for i in range(height):
        for j in range(width):
            region = padded_image[i:i+kernel_size, j:j+kernel_size]
            filtered_image[i, j] = np.max(region)

    return filtered_image

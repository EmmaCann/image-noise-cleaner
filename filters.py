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
# Tyson E, Jerry C, Colton P
# 11/11/2024
# Data Structures Project - AI Detection

import cv2
import numpy as np

class NoiseVisualization:

    def generate_noise_visualization(self, image):

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        f_transform = np.fft.fft2(gray_image)   # apply Fourier Transform
        f_shift = np.fft.fftshift(f_transform)

        # Create a high-pass filter to remove low frequencies:

        rows, cols = gray_image.shape
        crow, ccol = rows // 2, cols // 2 # Center
        mask = np.ones((rows, cols), np.uint8)

        r = 50 # Radius of low-frequency area to supress

        mask[crow - r:crow + r, ccol - r:ccol + r] = 0
        filtered_f_shift = f_shift * mask

        # Inverse Fourier Transform. This will be to visualize noise:

        f_shift = np.fft.ifftshift(filtered_f_shift)
        noise_image = np.fft.ifft2(f_shift)
        noise_image = np.abs(noise_image)

        normalized_noise = cv2.normalize(noise_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        normalized_noise = normalized_noise.astype(np.uint8)  # Convert to uint8 for visualization
        return normalized_noise


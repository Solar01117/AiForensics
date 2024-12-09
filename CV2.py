# Tyson Eifert
# 11/11/2024
# Data Structures Project - AI Detection
import tkinter

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from noise_visualization import NoiseVisualization

class ComputerVisionModule:

    def __init__(self, master):
        self.master = master
        master.title("AI Forensics")
        master.geometry("800x650")

        # Creating four quadrants for the application window:
        self.frame_top_left = tk.Frame(master, width=400, height=300, borderwidth=1, relief="solid")
        self.frame_top_left.grid(row=0, column=0, sticky="nsew")
        self.frame_top_right = tk.Frame(master, width=400, height=300, borderwidth=1, relief="solid")
        self.frame_top_right.grid(row=0, column=1, sticky="nsew")
        self.frame_bottom_left = tk.Frame(master, width=400, height=300, borderwidth=1, relief="solid")
        self.frame_bottom_left.grid(row=1, column=0, sticky="nsew")
        self.frame_bottom_right = tk.Frame(master, width=400, height=300, borderwidth=1, relief="solid")
        self.frame_bottom_right.grid(row=1, column=1, sticky="nsew")

        # Reconfigure the quadrants:
        master.grid_rowconfigure(0, weight=1)
        master.grid_rowconfigure(1, weight=1)
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)

        # Load button
        self.load_button = tk.Button(self.frame_top_left, text="Choose Image", command=self.load_image)
        self.load_button.pack(pady=20)

        # Quit button
        self.quit_button = tk.Button(self.frame_top_left, text="Quit", command=master.quit)
        self.quit_button.pack(pady=30)

        # Predict button
        self.predict_button = tk.Button(self.frame_top_left, text="Run Prediction Model",
                                        command=self.run_prediction_model)
        self.predict_button.pack(pady=10)
        self.predict_button.config(state=tk.DISABLED)  # Disabled until an image is chosen

        self.color_dist_button = tk.Button(self.frame_top_left, text="Color Distribution",
                                           command=self.run_color_distribution_analysis)
        self.color_dist_button.pack(pady=10)
        self.color_dist_button.config(state=tk.DISABLED)


        # Top right input image:
        self.image_canvas = tk.Canvas(self.frame_top_right, width=375, height=375)
        self.image_canvas.pack()

        # Bottom left image
        self.bottom_left_canvas = tk.Canvas(self.frame_bottom_left, width=375, height=375)
        self.bottom_left_canvas.pack()

        # Bottom right image (noise image)
        self.noise_canvas = tk.Canvas(self.frame_bottom_right, width=375, height=375)
        self.noise_canvas.pack()

        self.file_path = None  # Store the selected file path
        self.analysis_label_main = tk.Label(self.frame_top_left, text="Analysis Results: Waiting for input...",
                                            font=("Arial", 12))
        self.analysis_label_main.pack(pady=20)

        self.analysis_label_plot = tk.Label(self.frame_top_left, text="", font=("Arial", 12))
        self.analysis_label_plot.pack(pady=20)  # This label will display analysis result below the plot

        self.noise_visualizer = NoiseVisualization()
        self.noise_image = None
        self.image = None

    def load_image(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg"), ("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg")]
        )
        if self.file_path:
            # Load the image using OpenCV
            image = cv2.imread(self.file_path)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Store the image in the class attribute
            self.image = image  # This ensures the image is available for other methods

            # Resize and display the input image in top right quadrant
            pil_image = Image.fromarray(rgb_image)
            pil_image.thumbnail((300, 300))
            photo = ImageTk.PhotoImage(pil_image)
            self.image_canvas.create_image(0, 0, anchor="nw", image=photo)
            self.image_canvas.image = photo

            # Convert to grayscale and display in bottom left quadrant
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pil_gray_image = Image.fromarray(gray_image)
            pil_gray_image.thumbnail((300, 300))
            gray_photo = ImageTk.PhotoImage(pil_gray_image)
            self.bottom_left_canvas.create_image(0, 0, anchor="nw", image=gray_photo)
            self.bottom_left_canvas.image = gray_photo

            # Generate and display the noise visualization in bottom right quadrant
            self.noise_image = self.noise_visualizer.generate_noise_visualization(image)
            pil_noise_image = Image.fromarray(self.noise_image)
            pil_noise_image.thumbnail((300, 300))
            noise_photo = ImageTk.PhotoImage(pil_noise_image)
            self.noise_canvas.create_image(0, 0, anchor="nw", image=noise_photo)
            self.noise_canvas.image = noise_photo

            # Update the analysis label
            self.analysis_label_main.config(text="Noise Analysis: Visualized in bottom right quadrant.")

            # Enable the prediction button and color distribution button
            self.predict_button.config(state=tk.NORMAL)
            self.color_dist_button.config(state=tk.NORMAL)  # Enable the button

    def run_prediction_model(self):
        if self.file_path:  # Check if an image is loaded
            self.prediction_model(self.file_path)  # Pass the file path to prediction model

    def run_color_distribution_analysis(self):
        if self.image is not None:
            self.analyze_color_dist(self.image)

    def prediction_model(self, image_path):
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform FFT and compute the magnitude and phase spectrum
        f_transform = np.fft.fft2(gray_image)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        phase_spectrum = np.angle(f_shift)

        # Analyze high-frequency anomalies
        self.analyze_high_frequency_anomalies(magnitude_spectrum, phase_spectrum)

    def analyze_high_frequency_anomalies(self, magnitude_spectrum, phase_spectrum):
        """
        Detects high-frequency anomalies typical of AI-generated images.
        """
        # Mask high frequencies (consider frequencies above a threshold)
        rows, cols = magnitude_spectrum.shape
        crow, ccol = rows // 2, cols // 2  # center of the image

        # Filter out low frequencies (focus on high frequencies)
        high_freq_mask = np.zeros_like(magnitude_spectrum)
        high_freq_mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1  # central low frequencies
        high_freq_data = magnitude_spectrum * (1 - high_freq_mask)

        # Compute the "unnatural" high frequency anomalies
        unnatural_freqs = np.sum(high_freq_data)  # Sum of high-frequency components

        # Create a new Tkinter window for the plot
        plot_window = tk.Toplevel(self.master)
        plot_window.title("Frequency Spectrum Analysis")
        plot_window.geometry("800x600")

        # Create a label to display analysis result in the plot window
        analysis_label = tk.Label(plot_window, text=f"Unnatural Frequencies Sum: {unnatural_freqs:.2f}",
                                  font=("Arial", 12))
        analysis_label.pack(pady=10)  # Pack the label at the top of the window

        # Create two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the magnitude spectrum
        ax1.imshow(np.log(1 + high_freq_data), cmap='gray')
        ax1.set_title("Magnitude Spectrum - High Frequency Anomalies")
        ax1.set_xlabel("Frequency (pixels)")
        ax1.set_ylabel("Frequency (pixels)")
        ax1.grid(True)

        # Plot the phase spectrum
        ax2.imshow(phase_spectrum, cmap='gray')
        ax2.set_title("Phase Spectrum")
        ax2.set_xlabel("Frequency (pixels)")
        ax2.set_ylabel("Frequency (pixels)")
        ax2.grid(True)

        # Embed the plot into the Tkinter window using FigureCanvasTkAgg
        canvas = FigureCanvasTkAgg(fig, master=plot_window)  # Create canvas
        canvas.draw()  # Draw the figure
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def analyze_color_dist(self, image):

        r_channel, g_channel, b_channel = cv2.split(image)

        hist_r = cv2.calcHist([r_channel], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g_channel], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([b_channel], [0], None, [256], [0, 256])

        color_window = tk.Toplevel(self.master)
        color_window.title("Color Distribution Analysis")
        color_window.geometry("800x750")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(hist_r, color='red', label='Red Channel')
        ax.plot(hist_g, color='green', label='Green Channel')
        ax.plot(hist_b, color='blue', label='Blue Channel')
        ax.set_title("Color Distribution (RGB Histogram)")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True)

        # Embed the plot into the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=color_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
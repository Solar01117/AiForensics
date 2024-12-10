# Tyson Eifert
# 11/11/2024
# Data Structures Project - AI Detection
import datetime
import os
import psutil
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from noise_visualization import NoiseVisualization
import time
import math
from scipy.signal import convolve2d


class ComputerVisionModule:

    @staticmethod
    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss

    # decorator function
    def profile(func):
        def wrapper(*args, **kwargs):
            mem_before = ComputerVisionModule.process_memory()
            result = func(*args, **kwargs)
            mem_after = ComputerVisionModule.process_memory()
            print("{}:consumed memory: {:,}".format(
                func.__name__,
                mem_before, mem_after, mem_after - mem_before))

            return result

        return wrapper

    def timeme(method):
        def wrapper(args, **kw):
            start_time = int(round(time.time() * 1000))
            result = method(args, **kw)
            end_time = int(round(time.time() * 1000))
            print(end_time - start_time, 'ms')
            return result

        return wrapper


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

        # Noise Variance Estimation
        self.noise_variance_button = tk.Button(self.frame_top_left, text="Run Noise Variance Estimation",
                                               command=self.run_noise_variance_estimation)
        self.noise_variance_button.pack(pady=0)
        self.noise_variance_button.config(state=tk.DISABLED)  # Disabled until an image is chosen

        # Top right input image:
        self.image_canvas = tk.Canvas(self.frame_top_right, width=375, height=375)
        self.image_canvas.pack()

        # Bottom left image
        self.bottom_left_canvas = tk.Canvas(self.frame_bottom_left, width=375, height=375)
        self.bottom_left_canvas.pack()

        # Bottom right image (noise image)
        self.noise_canvas = tk.Canvas(self.frame_bottom_right, width=375, height=375)
        self.noise_canvas.pack()
        self.noise_variance_label_main = tk.Label(self.frame_top_left, text="Noise Variance Results: Waiting for "
                                                                            "input...", font=("Arial", 12))
        self.noise_variance_label_main.pack(pady=0)

        self.file_path = None  # Store the selected file path
        self.analysis_label_main = tk.Label(self.frame_top_left, text="Analysis Results: Waiting for input...",
                                            font=("Arial", 12))
        self.analysis_label_main.pack(pady=20)

        self.analysis_label_plot = tk.Label(self.frame_top_left, text="", font=("Arial", 12))
        self.analysis_label_plot.pack(pady=20)  # This label will display analysis result below the plot

        self.noise_visualizer = NoiseVisualization()
        self.noise_image = None

    @timeme
    @profile
    # The time me goes off how log the "file dialog is open", if you want to test times
    # just make sure to enter a file path yourself, the time this runs in is a little bit
    # faster and runs in around 40 milliseconds.
    def load_image(self):
        self.file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png;*.jpg;*. jpeg"), ("PNG files", "*.png"),
                        ("JPEG files", "*.jpg;*.jpeg")]
        )
        if self.file_path:
            # Load the image using OpenCV
            image = cv2.imread(self.file_path)

            # Convert to RGB and Grayscale once
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Function to handle image resizing and display in canvas
            def display_image(canvas, image_array, anchor="nw"):
                pil_image = Image.fromarray(image_array)
                pil_image.thumbnail((300, 300))
                photo = ImageTk.PhotoImage(pil_image)
                canvas.create_image(0, 0, anchor=anchor, image=photo)
                canvas.image = photo

            # Resize and display the original RGB image in top right quadrant
            display_image(self.image_canvas, rgb_image)

            # Display the grayscale image in bottom left quadrant
            display_image(self.bottom_left_canvas, gray_image)

            # Generate and display the noise visualization in bottom right quadrant
            self.noise_image = self.noise_visualizer.generate_noise_visualization(image)
            display_image(self.noise_canvas, self.noise_image)

            # Update the analysis label and enable the prediction button
            self.analysis_label_main.config(text="Noise Analysis: Visualized in bottom right quadrant.")
            self.predict_button.config(state=tk.NORMAL)

            # Update the analysis label to enable the noise variance estimation button
            self.noise_variance_button.config(state=tk.NORMAL)
    @timeme
    @profile
    def run_noise_variance_estimation(self):
        if self.file_path:
            self.noise_variance_estimation(self.file_path)

    @timeme
    @profile
    def run_prediction_model(self):
        if self.file_path:  # Check if an image is loaded
            self.prediction_model(self.file_path)  # Pass the file path to prediction model

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

    def noise_variance_estimation(self, img_path):
        # Reads the file path and gives the array for the image
        image = cv2.imread(img_path)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        H, W = img_gray.shape

        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]

        np_M = np.array(M)  # Changing this array to a numpy array changed the speed from 1 second to .04 seconds!

        # Equation to figure out the noise variation of the image
        sigma = np.sum(np.sum(np.absolute(convolve2d(img_gray, np_M))))
        sigma = sigma * math.sqrt(0.5 * math.pi) / (6 * (W - 2) * (H - 2))

        self.noise_variance_label_main.config(text="Noise Analysis Done: " + str(sigma))
        return sigma

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

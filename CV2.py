# Tyson E, Jerry C, Colton P
# 11/11/2024
# Data Structures Project - AI Detection

from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from noise_visualization import NoiseVisualization
import os
import psutil
import time
import math
from scipy.signal import convolve2d


class ComputerVisionModule:

    @staticmethod
    def process_memory():
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss

    # Memory profiling decorator
    def profile(func):
        def wrapper(*args, **kwargs):
            mem_before = ComputerVisionModule.process_memory()
            result = func(*args, **kwargs)
            mem_after = ComputerVisionModule.process_memory()
            print(f"{func.__name__}: Consumed memory: {mem_after - mem_before:,} bytes")
            return result

        return wrapper

    # Timing decorator with input display and time taken in a temporary Tkinter window
    def timeme(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            # Create a Tkinter window to display function inputs and time
            temp_window = tk.Toplevel()
            temp_window.title(f"Running: {func.__name__}")
            temp_window.geometry("400x300")  # Adjusted window size for clarity

            input_label = tk.Label(temp_window, text=f"Inputs and running time "
                                                     f"for {func.__name__}:", font=("Arial", 12))
            input_label.pack(pady=10)

            result = func(*args, **kwargs)
            end_time = time.time()

            # Calculate time taken
            time_taken = (end_time - start_time) * 1000  # in milliseconds

            # Display time taken in the window
            time_label = tk.Label(temp_window, text=f"Time taken: {time_taken:.2f} ms", font=("Arial", 12))
            time_label.pack(pady=10)

            return result

        return wrapper

    def __init__(self, master):
        self.master = master
        master.title("AI Forensics")
        master.geometry("800x680")

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

        self.text_label = tk.Label(self.frame_top_left, text="For protection: This program does not store or transmit "
                                                             "any data by default. Analytics are stored by the user.",
                                   wraplength=350, justify='center', anchor='center')
        self.text_label.pack(pady=10)

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

        # Noise variance button (button for variance)
        self.variance_button = tk.Button(self.frame_top_left, text="Estimate Noise Variance",
                                               command=self.estimate_noise_variance)
        self.variance_button.pack(pady=10)
        self.variance_button.config(state=tk.DISABLED)  # Disabled until an image is loaded

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

        # Add the noise variance label here
        self.noise_variance_label_main = tk.Label(self.frame_top_left, text="Noise Analysis: Waiting for input...",
                                                  font=("Arial", 12))
        self.noise_variance_label_main.pack(pady=10)  # Adjust the padding as necessary

        self.noise_visualizer = NoiseVisualization()
        self.noise_image = None
        self.image = None

    @timeme
    def load_image(self):
        try:
            # Ask for file and check if it's selected
            self.file_path = filedialog.askopenfilename(
                filetypes=[("Image Files", "*.png;*.jpg;*.jpeg"), ("PNG files", "*.png"),
                          ("JPEG files", "*.jpg;*.jpeg")]
            )
            if not self.file_path:
                raise ValueError("No file selected.")

            # Load the image with OpenCV
            image = cv2.imread(self.file_path)
            if image is None:
                raise ValueError("Failed to load image. Please check the file format.")

            # Convert image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Store the image for further processing
            self.image = image

            # Optimize resizing by resizing to a fixed size directly using PIL (no aspect ratio constraint)
            pil_image = Image.fromarray(rgb_image)
            pil_image = pil_image.resize((300, 300), Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality downsampling
            photo = ImageTk.PhotoImage(pil_image)

            # Display the resized image on the canvas (top-right quadrant)
            self.image_canvas.create_image(0, 0, anchor="nw", image=photo)
            self.image_canvas.image = photo  # Keep a reference to avoid garbage collection

            # Convert to grayscale and display in bottom-left quadrant
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            pil_gray_image = Image.fromarray(gray_image)
            pil_gray_image = pil_gray_image.resize((300, 300), Image.Resampling.LANCZOS)  # Use LANCZOS
            gray_photo = ImageTk.PhotoImage(pil_gray_image)
            self.bottom_left_canvas.create_image(0, 0, anchor="nw", image=gray_photo)
            self.bottom_left_canvas.image = gray_photo  # Keep a reference

            # Generate noise visualization and display in bottom-right quadrant
            self.noise_image = self.noise_visualizer.generate_noise_visualization(image)
            pil_noise_image = Image.fromarray(self.noise_image)
            pil_noise_image = pil_noise_image.resize((300, 300), Image.Resampling.LANCZOS)  # Use LANCZOS
            noise_photo = ImageTk.PhotoImage(pil_noise_image)
            self.noise_canvas.create_image(0, 0, anchor="nw", image=noise_photo)
            self.noise_canvas.image = noise_photo  # Keep a reference

            # Update the analysis label
            self.analysis_label_main.config(text="Noise Analysis: Visualized in bottom right quadrant.")

            # Enable prediction and color distribution buttons
            self.predict_button.config(state=tk.NORMAL)
            self.color_dist_button.config(state=tk.NORMAL)
            self.variance_button.config(state=tk.NORMAL)

        except Exception as e:
            self.show_error(f"Error loading image: {str(e)}")

    def show_error(self, error_message):
        error_window = tk.Toplevel(self.master)
        error_window.title("Error")
        error_window.geometry("400x200")

        error_label = tk.Label(error_window, text=error_message, font=("Arial", 12), fg="red", wraplength=350)
        error_label.pack(pady=20)

        ok_button = tk.Button(error_window, text="OK", command=error_window.destroy)
        ok_button.pack(pady=10)

    def estimate_noise_variance(self):
        if self.file_path:
            try:
                # Call the noise variance estimation method
                sigma = self.noise_variance_estimation(self.file_path)

                # Open a new window to display the variance
                variance_window = tk.Toplevel(self.master)
                variance_window.title("Noise Variance")
                variance_window.geometry("450x250")  # Adjust size to fit the label and tips

                # Create a label to display the calculated noise variance
                variance_label = tk.Label(variance_window, text=f"Noise Variance: {sigma:.4f}", font=("Arial", 12))
                variance_label.pack(pady=20)

                # Create a tip label to guide users on interpreting the variance values
                tips_label = tk.Label(variance_window, text="Tips:\n\n"
                                                            "• Low variance (~0): Image has minimal noise.\n"
                                                            "• Moderate variance (<1): Image has noticeable noise.\n"
                                                            "• High variance (>1): Image is highly noisy or may be artificially generated.",
                                      font=("Arial", 10), justify="center", anchor="w")
                tips_label.pack(pady=10, padx=10)

                # Optionally add a close button to the new window
                close_button = tk.Button(variance_window, text="Close", command=variance_window.destroy)
                close_button.pack(pady=10)

            except Exception as e:
                self.show_error(f"Error estimating noise variance: {str(e)}")

    @timeme
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

        return sigma

    def run_prediction_model(self):
        if self.file_path:  # Check if an image is loaded
            self.prediction_model(self.file_path)  # Pass the file path to prediction model

    def run_color_distribution_analysis(self):
        if self.image is not None:
            self.analyze_color_dist(self.image)

    @timeme
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

    @timeme
    def analyze_color_dist(self, image):

        r_channel, g_channel, b_channel = cv2.split(image)

        hist_r = cv2.calcHist([r_channel], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([g_channel], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([b_channel], [0], None, [256], [0, 256])

        color_window = tk.Toplevel(self.master)
        color_window.title("Color Distribution Analysis")
        color_window.geometry("800x750")

        # Create a frame to hold both the time information and the plot
        frame = tk.Frame(color_window)
        frame.pack(pady=20)

        # Create a label to display time taken
        time_label = tk.Label(frame, text="Running time: TBD", font=("Arial", 12))
        time_label.pack()

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

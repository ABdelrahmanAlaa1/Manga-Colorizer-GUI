import os
import sys
import argparse
import time
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.font import Font
import threading
import queue
import json
from collections import deque
import webbrowser
import random
import glob
from PIL import ImageTk

import PIL.Image
import numpy as np

# --- Path Setup (Adjusted for root directory) ---
# Get the directory of the current script (which is now the project root)
# and join it with 'Backend' to get the correct path.
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Backend'))
sys.path.insert(0, backend_path)

# --- Now we can import from Backend ---
from colorizator import MangaColorizator
from denoisator import MangaDenoiser
from upscalator import MangaUpscaler
from utils.utils import distance_from_grayscale, save_image, clear_torch_cache


# --- Core Processing Logic ---
def process_image(image_path, output_folder, colorizer, upscaler, denoiser, config, progress_queue, overwrite_existing):
    """
    Processes a single image: denoises, colorizes, and upscales based on the config.
    Sends progress updates back to the GUI via a queue.
    Returns the time taken to process the image.
    """
    try:
        image_name = os.path.basename(image_path)
        output_path = os.path.join(output_folder, image_name)

        if not overwrite_existing and os.path.exists(output_path):
            progress_queue.put({'type': 'log', 'message': f"[~] Skipping '{image_name}' as it already exists."})
            return 0 # Return 0 time taken for skipped files

        progress_queue.put({'type': 'log', 'message': f"\n--- Processing: {image_name} ---"})

        image = PIL.Image.open(image_path).convert("RGB")
        image = np.array(image)

        coloredness = distance_from_grayscale(image)
        if coloredness > 15 and config.colorize:
            progress_queue.put({'type': 'log', 'message': f"[!] '{image_name}' appears to be already colored. Skipping colorization step."})
            if not config.denoise and not config.upscale:
                return 0

        start_time = time.time()

        if config.denoise and denoiser:
            progress_queue.put({'type': 'log', 'message': "[*] Denoising..."})
            image = denoiser.denoise(image, config.denoise_sigma)

        if config.colorize and colorizer:
            progress_queue.put({'type': 'log', 'message': "[*] Colorizing..."})

            # This logic ensures the image width sent to the colorizer is a multiple of 32.
            original_width = image.shape[1]
            target_width = config.colorized_image_size

            # Use the smaller of the original width or the user-defined target width.
            effective_width = min(original_width, target_width)

            # Calculate a new width that is a multiple of 32 by rounding down.
            adjusted_width = effective_width - (effective_width % 32)

            # Handle cases where the image is extremely narrow (less than 32px wide).
            if adjusted_width == 0:
                adjusted_width = 32
                progress_queue.put({'type': 'log', 'message': f"[!] Warning: Image width is very small. Setting colorizer width to a minimum of {adjusted_width}px."})

            if adjusted_width != original_width:
                 progress_queue.put({'type': 'log', 'message': f"[*] Adjusting image width for colorizer: {original_width}px -> {adjusted_width}px"})

            # Set the image for the colorizer with the new, safe width.
            colorizer.set_image((image.astype('float32') / 255), adjusted_width)

            image = colorizer.colorize()

        if config.upscale and upscaler:
            progress_queue.put({'type': 'log', 'message': f"[*] Upscaling by {config.upscale_factor}x..."})
            image = upscaler.upscale((image.astype('float32') / 255), config.upscale_factor)

        save_image(image, output_path)

        end_time = time.time()
        duration = end_time - start_time
        progress_queue.put({'type': 'log', 'message': f"[+] Finished '{image_name}' in {duration:.2f}s."})
        return duration

    except Exception as e:
        progress_queue.put({'type': 'log', 'message': f"[!!!] FAILED to process {image_path}. Error: {e}"})
        import traceback
        print(f"--- ERROR TRACEBACK for {image_path} ---")
        traceback.print_exc()
        print("------------------------------------")
        return 0

# --- GUI Application ---
class ColorizerApp(tk.Tk):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.settings_file = "settings.json"

        self.title("Manga Colorizer")
        self.geometry("550x700")

        # --- Variables ---
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.enable_colorize = tk.BooleanVar(value=True)
        self.enable_upscale = tk.BooleanVar(value=True)
        self.enable_denoise = tk.BooleanVar(value=True)
        self.overwrite_existing = tk.BooleanVar(value=False)
        self.eta_text = tk.StringVar(value="ETA: N/A")

        # --- Advanced settings variables ---
        default_networks_path = os.path.join(backend_path, 'networks')
        self.colorizer_path = tk.StringVar(value=os.path.join(default_networks_path, 'generator.zip'))
        self.upscaler_path = tk.StringVar(value=os.path.join(default_networks_path, 'RealESRGAN_x4plus_anime_6B.pt'))
        self.upscaler_type = tk.StringVar(value='ESRGAN')
        self.denoise_sigma = tk.IntVar(value=25)
        self.upscaler_tile_size = tk.IntVar(value=256)
        self.colorizer_tile_size = tk.IntVar(value=0)
        self.tile_pad = tk.IntVar(value=8)
        self.colorized_image_size = tk.IntVar(value=576)


        self.processing_thread = None
        self.terminate_event = threading.Event() # Event to signal thread termination on window close
        self.pause_event = threading.Event() # Using a separate event for pausing

        # --- Model Management ---
        self.colorizer_model = None
        self.upscaler_model = None
        self.denoiser_model = None

        # --- State Tracking ---
        self.current_input_path = None


        # --- Style for labels ---
        self.style = ttk.Style(self)
        italic_font = Font(family="Helvetica", size=10, slant="italic")
        self.style.configure("Italic.TLabel", font=italic_font)

        # --- Style for hyperlink ---
        link_font = Font(family="Helvetica", size=11, underline=True)
        self.style.configure("Link.TLabel", foreground="blue", font=link_font)


        self.create_widgets()
        self.load_settings()

        self.progress_queue = queue.Queue()
        self.after(100, self.check_queue)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        folder_frame = ttk.LabelFrame(main_frame, text="Folders", padding="10")
        folder_frame.pack(fill=tk.X, pady=5)
        folder_frame.columnconfigure(1, weight=1)
        ttk.Label(folder_frame, text="Input Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(folder_frame, textvariable=self.input_folder).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse...", command=self.select_input_folder).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(folder_frame, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(folder_frame, textvariable=self.output_folder).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse...", command=self.select_output_folder).grid(row=1, column=2, padx=5, pady=5)

        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(options_frame, text="Enable Colorizer", variable=self.enable_colorize).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Enable Upscaling", variable=self.enable_upscale).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(options_frame, text="Enable Denoising", variable=self.enable_denoise).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Overwrite existing files", variable=self.overwrite_existing).grid(row=1, column=1, sticky="w")

        advanced_button = ttk.Button(options_frame, text="Advanced Settings...", command=self.open_advanced_settings)
        advanced_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10,0))

        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        progress_frame.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0,10))
        ttk.Label(progress_frame, textvariable=self.eta_text).grid(row=0, column=1, sticky="e")

        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.pause_button = ttk.Button(action_frame, text="Pause", command=self.pause_processing, state="disabled")
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar.set)

        # --- Bottom bar for links and buttons ---
        bottom_bar_frame = ttk.Frame(log_frame)
        bottom_bar_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5,0))
        bottom_bar_frame.columnconfigure(1, weight=1) # Make column 1 (where button is) expand to push button right

        # Bug report link
        bug_report_label = ttk.Label(bottom_bar_frame, text="Found a bug? Report @sepTN", style="Link.TLabel", cursor="hand2")
        bug_report_label.grid(row=0, column=0, sticky="w", padx=5) # Align to the west (left)
        bug_report_label.bind("<Button-1>", self.open_bug_report_link)

        # --- Button container for the right side ---
        button_container = ttk.Frame(bottom_bar_frame)
        button_container.grid(row=0, column=1, sticky="e")

        # Preview button
        preview_button = ttk.Button(button_container, text="Preview", command=self.open_preview_window)
        preview_button.pack(side=tk.LEFT, padx=(0, 5))

        # Clear log button
        clear_button = ttk.Button(button_container, text="Clear Log", command=self.clear_log)
        clear_button.pack(side=tk.LEFT)


    def open_preview_window(self):
        # --- 1. Get the user-selected input folder and images ---
        image_folder = self.input_folder.get().strip().strip("'\"")
        if not image_folder or not os.path.isdir(image_folder):
            self.log("[!] Please select a valid input folder first to use the preview.")
            return

        supported_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith(supported_ext)]

        if not image_paths:
            self.log(f"[!] No images found in '{os.path.basename(image_folder)}' for preview.")
            return

        # --- 2. Create the window ---
        preview_window = tk.Toplevel(self)
        preview_window.title("Colorizer Preview")
        preview_window.transient(self)
        preview_window.grab_set()

        main_frame = ttk.Frame(preview_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1) # Allow image labels to expand

        # --- 3. Create Widgets ---
        ttk.Label(main_frame, text="Original", font=("Helvetica", 12, "bold")).grid(row=0, column=0, pady=(0, 5))
        ttk.Label(main_frame, text="Colorized", font=("Helvetica", 12, "bold")).grid(row=0, column=1, pady=(0, 5))

        original_label = ttk.Label(main_frame)
        original_label.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        colorized_label = ttk.Label(main_frame)
        colorized_label.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        repick_button = ttk.Button(main_frame, text="Repick")
        repick_button.grid(row=2, column=0, columnspan=2, pady=(10, 0))

        # --- 4. Define the core worker function ---
        def load_and_colorize_image(image_path):
            original_cwd = os.getcwd()
            try:
                # Disable button and set loading text (via main thread)
                def set_loading_state():
                    repick_button.config(state="disabled")
                    original_label.config(image='', text="Loading...")
                    colorized_label.config(image='', text="Loading...")
                self.after(0, set_loading_state)

                # --- Actual work in thread ---
                os.chdir(backend_path)
                self._update_config_from_gui()
                colorizer, _, denoiser = self._get_and_manage_models()

                if not colorizer:
                    self.log("[!] Colorizer model could not be loaded.")
                    self.after(0, preview_window.destroy)
                    return

                original_image_pil = PIL.Image.open(image_path).convert("RGB")
                image_np = np.array(original_image_pil)

                # Denoise step
                if self.config.denoise and denoiser:
                    self.after(0, lambda: colorized_label.config(text="Denoising..."))
                    image_np = denoiser.denoise(image_np, self.config.denoise_sigma)

                self.after(0, lambda: colorized_label.config(text="Colorizing..."))

                target_width = self.config.colorized_image_size
                original_width = image_np.shape[1]
                effective_width = min(original_width, target_width)
                adjusted_width = effective_width - (effective_width % 32)
                if adjusted_width == 0: adjusted_width = 32

                colorizer.set_image((image_np.astype('float32') / 255), adjusted_width)
                colorized_image_np = colorizer.colorize()
                colorized_image_pil = PIL.Image.fromarray(colorized_image_np)

                max_size = (512, 768)
                original_image_pil.thumbnail(max_size, PIL.Image.Resampling.LANCZOS)
                colorized_image_pil.thumbnail(max_size, PIL.Image.Resampling.LANCZOS)

                original_photo = ImageTk.PhotoImage(original_image_pil)
                colorized_photo = ImageTk.PhotoImage(colorized_image_pil)

                # --- Update GUI in main thread ---
                def update_gui_images():
                    original_label.config(image=original_photo, text="")
                    original_label.image = original_photo
                    colorized_label.config(image=colorized_photo, text="")
                    colorized_label.image = colorized_photo
                self.after(0, update_gui_images)

            except Exception as e:
                self.log(f"[!!!] Failed to create preview for {os.path.basename(image_path)}. Error: {e}")
                self.after(0, preview_window.destroy)
            finally:
                os.chdir(original_cwd)
                # Re-enable button in main thread
                self.after(0, lambda: repick_button.config(state="normal"))

        # --- 5. Define the button command ---
        def start_new_pick():
            new_path = random.choice(image_paths)
            self.log(f"[*] Previewing random image: {os.path.basename(new_path)}")
            threading.Thread(target=lambda: load_and_colorize_image(new_path), daemon=True).start()

        repick_button.config(command=start_new_pick)
        start_new_pick() # Initial load

    def open_advanced_settings(self):
        adv_window = tk.Toplevel(self)
        adv_window.title("Advanced Settings")
        adv_window.geometry("550x700")
        adv_window.transient(self)
        adv_window.grab_set()

        main_frame = ttk.Frame(adv_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        paths_frame = ttk.LabelFrame(main_frame, text="Model Paths", padding="10")
        paths_frame.pack(fill=tk.X, pady=5)
        paths_frame.columnconfigure(1, weight=1)
        ttk.Label(paths_frame, text="Colorizer:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(paths_frame, textvariable=self.colorizer_path).grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(paths_frame, text="...", command=lambda: self.select_model_path(self.colorizer_path)).grid(row=0, column=2, padx=5, pady=2)
        ttk.Label(paths_frame, text="Upscaler:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Entry(paths_frame, textvariable=self.upscaler_path).grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        ttk.Button(paths_frame, text="...", command=lambda: self.select_model_path(self.upscaler_path)).grid(row=1, column=2, padx=5, pady=2)

        perf_frame = ttk.LabelFrame(main_frame, text="Performance", padding="10")
        perf_frame.pack(fill=tk.X, pady=5)

        def create_slider(parent, text, variable, from_, to, desc):
            frame = ttk.Frame(parent)
            label_text = f"{text} ({variable.get()})"
            label = ttk.Label(frame, text=label_text)
            label.pack(anchor="w")

            def update_label(val):
                label.config(text=f"{text} ({int(float(val))})")

            slider = ttk.Scale(frame, from_=from_, to=to, orient="horizontal", variable=variable, command=update_label)
            slider.pack(fill=tk.X, pady=(2,0))
            desc_label = ttk.Label(frame, text=desc, wraplength=450, justify="left", style="Italic.TLabel")
            desc_label.pack(anchor="w", pady=(0, 10))
            return frame

        create_slider(perf_frame, "Upscaler Tile Size", self.upscaler_tile_size, 0, 1024,
                      "Size of tiles (in pixels) for upscaling to save memory. 0 disables tiling. Default: 256").pack(fill=tk.X)
        create_slider(perf_frame, "Colorizer Tile Size", self.colorizer_tile_size, 0, 1024,
                      "Size of tiles for colorizing. Usually not needed. 0 disables tiling. Default: 0").pack(fill=tk.X)
        create_slider(perf_frame, "Tile Padding", self.tile_pad, 0, 64,
                      "Pixel overlap between tiles to reduce seams. Default: 8").pack(fill=tk.X)
        create_slider(perf_frame, "Colorized Image Size", self.colorized_image_size, 256, 1184,
                      "Target width for the colorizer. Larger isn't always better, but can help with blurry text. Default: 576").pack(fill=tk.X)

        upscaler_frame = ttk.LabelFrame(main_frame, text="Upscaler Type", padding="10")
        upscaler_frame.pack(fill=tk.X, pady=5)
        upscaler_combo = ttk.Combobox(upscaler_frame, textvariable=self.upscaler_type, values=['ESRGAN', 'GigaGAN'], state="readonly")
        upscaler_combo.pack(fill=tk.X)

        denoise_frame = ttk.LabelFrame(main_frame, text="Denoise Sigma", padding="10")
        denoise_frame.pack(fill=tk.X, pady=5)
        create_slider(denoise_frame, "Value", self.denoise_sigma, 1, 100, "Strength of the denoiser. Default: 25").pack(fill=tk.X)

        ttk.Button(main_frame, text="Close", command=adv_window.destroy).pack(side=tk.BOTTOM, pady=10)

    def open_bug_report_link(self, event):
        """Opens the GitHub issues page in a new browser tab."""
        webbrowser.open_new_tab("https://github.com/sepTN/Manga-Colorizer-GUI/issues")

    def select_model_path(self, path_var):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=(("Model files", "*.pt *.pth *.zip"), ("All files", "*.*"))
        )
        if file_path:
            path_var.set(file_path)

    def select_input_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.input_folder.set(folder_path)
            current_output = self.output_folder.get()
            if not current_output or current_output.startswith(os.path.dirname(self.input_folder.get())):
                 self.output_folder.set(os.path.join(folder_path, 'output'))

    def select_output_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_folder.set(folder_path)

    def log(self, message):
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")

    def check_queue(self):
        while not self.progress_queue.empty():
            data = self.progress_queue.get()
            if data['type'] == 'log':
                self.log(data['message'])
            elif data['type'] == 'progress':
                self.progress_bar['value'] = data['value']
                self.eta_text.set(data['eta'])
        self.after(100, self.check_queue)

    def _update_config_from_gui(self):
        """Updates the self.config object with the current values from the GUI widgets."""
        # Basic options
        self.config.colorize = self.enable_colorize.get()
        self.config.upscale = self.enable_upscale.get()
        self.config.denoise = self.enable_denoise.get()

        # Advanced settings
        self.config.colorizer_path = self.colorizer_path.get()
        self.config.upscaler_path = self.upscaler_path.get()
        self.config.upscaler_type = self.upscaler_type.get()
        self.config.denoise_sigma = self.denoise_sigma.get()
        self.config.upscaler_tile_size = self.upscaler_tile_size.get()
        self.config.colorizer_tile_size = self.colorizer_tile_size.get()
        self.config.tile_pad = self.tile_pad.get()
        self.config.colorized_image_size = self.colorized_image_size.get()

    def _get_and_manage_models(self):
        """
        Checks GUI toggles and loads/returns models as needed.
        This is called by the processing thread before each image.
        """
        try:
            # --- Colorizer ---
            if self.config.colorize and self.colorizer_model is None:
                self.progress_queue.put({'type': 'log', 'message': "[*] Colorizing enabled. Initializing Colorizer model..."})
                self.colorizer_model = MangaColorizator(self.config)

            # --- Upscaler ---
            if self.config.upscale and self.upscaler_model is None:
                self.progress_queue.put({'type': 'log', 'message': "[*] Upscaling enabled. Initializing Upscaler model..."})
                self.upscaler_model = MangaUpscaler(self.config)

            # --- Denoiser ---
            if self.config.denoise and self.denoiser_model is None:
                self.progress_queue.put({'type': 'log', 'message': "[*] Denoising enabled. Initializing Denoiser model..."})
                self.denoiser_model = MangaDenoiser(self.config)

            # Return the currently active models based on toggles
            active_colorizer = self.colorizer_model if self.config.colorize else None
            active_upscaler = self.upscaler_model if self.config.upscale else None
            active_denoiser = self.denoiser_model if self.config.denoise else None

            return active_colorizer, active_upscaler, active_denoiser

        except Exception as e:
            self.progress_queue.put({'type': 'log', 'message': f"[!!!] FAILED to initialize a model. Error: {e}"})
            # Return None for all models to stop processing for the current image
            return None, None, None


    def start_processing(self):
        # Check if another process is already running
        if self.processing_thread and self.processing_thread.is_alive():
            self.log("[!] A process is already running. Please wait or pause it.")
            return

        input_path = self.input_folder.get().strip().strip("'\"")
        output_path = self.output_folder.get().strip().strip("'\"")

        if not input_path or not output_path:
            self.log("[!] Please select both input and output folders.")
            return
        if not os.path.isdir(input_path):
            self.log(f"[!] Error: Input folder not found at '{input_path}'")
            return

        os.makedirs(output_path, exist_ok=True)
        
        # Store the input path for the new process
        self.current_input_path = input_path

        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal", text="Pause", command=self.pause_processing)
        self.terminate_event.clear()
        self.pause_event.clear()

        # Reset progress bar for a new run
        self.progress_bar['value'] = 0
        self.eta_text.set("ETA: N/A")

        self.processing_thread = threading.Thread(
            target=self.run_batch_processing,
            args=(input_path, output_path),
            daemon=True
        )
        self.processing_thread.start()

    def pause_processing(self):
        self.log("\n[!] Pausing... will pause after the current file is finished.")
        self.pause_event.set()
        self.pause_button.config(text="Continue", command=self.continue_processing)

    def continue_processing(self):
        new_input_path = self.input_folder.get().strip().strip("'\"")

        # Check if the input folder has changed, requiring a full restart
        if new_input_path != self.current_input_path:
            self.log("\n[!] Input folder has changed. Restarting the entire batch process...")

            # Signal the current thread to stop.
            self.terminate_event.set()
            # Un-pause the thread so it can see the terminate signal and exit its loop.
            self.pause_event.clear()

            def restart_after_termination():
                if self.processing_thread and self.processing_thread.is_alive():
                    # Check again in 100ms
                    self.after(100, restart_after_termination)
                else:
                    # The old thread is gone, now we can start a new process.
                    self.start_processing()

            # Start the check to wait for the old thread to terminate.
            self.after(100, restart_after_termination)
        else:
            # Path is the same, just resume.
            self.log("\n[*] Resuming...")
            self.pause_event.clear()
            self.pause_button.config(text="Pause", command=self.pause_processing)

    def run_batch_processing(self, input_path, output_path):
        original_cwd = os.getcwd()
        os.chdir(backend_path) # Change directory for the duration of the thread

        supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        images_to_process = sorted([f for f in os.listdir(input_path) if f.lower().endswith(supported_formats)])
        total_images = len(images_to_process)

        if not images_to_process:
            self.progress_queue.put({'type': 'log', 'message': f"[!] No supported image files found in '{input_path}'."})
            os.chdir(original_cwd)
            self.processing_finished()
            return

        self.progress_queue.put({'type': 'log', 'message': f"\n[*] Found {total_images} images to process."})

        recent_times = deque(maxlen=10)

        for i, image_file in enumerate(images_to_process):
            if self.terminate_event.is_set():
                break

            # This is the pause loop
            while self.pause_event.is_set():
                if self.terminate_event.is_set():
                    break
                time.sleep(0.2)
            if self.terminate_event.is_set():
                break

            # Update config and get active models for every image
            self._update_config_from_gui()
            colorizer, upscaler, denoiser = self._get_and_manage_models()

            # If a model failed to load, _get_and_manage_models returns Nones and logs an error.
            # We can skip to the next image.
            if self.config.colorize and colorizer is None: continue
            if self.config.upscale and upscaler is None: continue
            if self.config.denoise and denoiser is None: continue

            overwrite = self.overwrite_existing.get()
            full_image_path = os.path.join(input_path, image_file)

            duration = process_image(full_image_path, output_path, colorizer, upscaler, denoiser, self.config, self.progress_queue, overwrite)

            if duration > 0:
                recent_times.append(duration)

            progress_percent = ((i + 1) / total_images) * 100

            eta_string = "ETA: Calculating..."
            if recent_times:
                avg_time = sum(recent_times) / len(recent_times)
                remaining_images = total_images - (i + 1)
                eta_seconds = int(avg_time * remaining_images)
                eta_string = f"ETA: {time.strftime('%H:%M:%S', time.gmtime(eta_seconds))}"

            self.progress_queue.put({'type': 'progress', 'value': progress_percent, 'eta': eta_string})

        # --- Cleanup after loop ---
        os.chdir(original_cwd) # Change back to original directory
        clear_torch_cache()

        if self.terminate_event.is_set():
             self.progress_queue.put({'type': 'log', 'message': "\n--- Processing terminated by user. ---"})
        else:
            self.progress_queue.put({'type': 'log', 'message': "\n--- Batch processing complete! ---"})
            self.progress_queue.put({'type': 'progress', 'value': 100, 'eta': 'ETA: Done!'})

        self.processing_finished()

    def processing_finished(self):
        self.start_button.config(state="normal")
        self.pause_button.config(state="disabled", text="Pause") # Reset button text
        self.processing_thread = None


    def save_settings(self):
        settings = {
            "input_folder": self.input_folder.get(),
            "output_folder": self.output_folder.get(),
            "enable_colorize": self.enable_colorize.get(),
            "enable_upscale": self.enable_upscale.get(),
            "enable_denoise": self.enable_denoise.get(),
            "overwrite_existing": self.overwrite_existing.get(),
            "denoise_sigma": self.denoise_sigma.get(),
            "colorizer_path": self.colorizer_path.get(),
            "upscaler_path": self.upscaler_path.get(),
            "upscaler_type": self.upscaler_type.get(),
            "upscaler_tile_size": self.upscaler_tile_size.get(),
            "colorizer_tile_size": self.colorizer_tile_size.get(),
            "tile_pad": self.tile_pad.get(),
            "colorized_image_size": self.colorized_image_size.get()
        }
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.log(f"[!] Could not save settings: {e}")

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.input_folder.set(settings.get("input_folder", ""))
                    self.output_folder.set(settings.get("output_folder", ""))
                    self.enable_colorize.set(settings.get("enable_colorize", True))
                    self.enable_upscale.set(settings.get("enable_upscale", True))
                    self.enable_denoise.set(settings.get("enable_denoise", True))
                    self.overwrite_existing.set(settings.get("overwrite_existing", False))
                    self.denoise_sigma.set(settings.get("denoise_sigma", 25))

                    default_networks_path = os.path.join(backend_path, 'networks')
                    self.colorizer_path.set(settings.get("colorizer_path", os.path.join(default_networks_path, 'generator.zip')))
                    self.upscaler_path.set(settings.get("upscaler_path", os.path.join(default_networks_path, 'RealESRGAN_x4plus_anime_6B.pt')))
                    self.upscaler_type.set(settings.get("upscaler_type", 'ESRGAN'))

                    self.upscaler_tile_size.set(settings.get("upscaler_tile_size", 256))
                    self.colorizer_tile_size.set(settings.get("colorizer_tile_size", 0))
                    self.tile_pad.set(settings.get("tile_pad", 8))
                    self.colorized_image_size.set(settings.get("colorized_image_size", 576))

        except Exception as e:
            self.log(f"[!] Could not load settings: {e}")

    def on_closing(self):
        self.terminate_event.set() # Signal thread to stop
        self.save_settings()
        if self.processing_thread and self.processing_thread.is_alive():
             self.log("[!] Waiting for current file to finish before closing...")
             # This is a simple wait, a more robust solution might use thread.join with a timeout
             self.after(100, self.check_thread_and_close)
        else:
             self.destroy()

    def check_thread_and_close(self):
        if self.processing_thread and self.processing_thread.is_alive():
            self.after(100, self.check_thread_and_close)
        else:
            self.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standalone Manga Colorizer')
    # Set a default for the device based on the OS.
    default_device = 'cuda'
    if sys.platform == 'darwin':
        default_device = 'mps'
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=default_device, help='Device to use for processing')

    # These arguments are now primarily for setting initial defaults,
    # as the GUI settings will override them.
    config = parser.parse_args()

    # These are now just fallback defaults if settings.json is missing/corrupt
    config.upscaler_tile_size = 256
    config.colorizer_tile_size = 0
    config.tile_pad = 8
    config.colorized_image_size = 576
    config.upscale_factor = 4 # This is not currently user-configurable

    app = ColorizerApp(config)
    app.mainloop()

# ui/settings_panel.py
import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, simpledialog, messagebox

from ui.utils import HoverTooltip

def open_advanced_settings(self):
    from pipeline import (
        INSTANCE_CAPS,
        THREADS_PER_INSTANCE,
        calculate_max_instances_from_vram,
        estimate_vram_breakdown_mb,
        get_default_instance_vram_profile_mb,
        get_cuda_memory_stats_mb,
        get_system_memory_stats_mb,
        get_gpu_vram_mb,
        vram_warning_text,
    )
    from precision_utils import (
        supports_fp8,
        POLICY_FP8,
        normalize_precision_policy,
        normalize_fallback_mode,
    )

    if getattr(self, 'advanced_window_ref', None) and self.advanced_window_ref.winfo_exists():
        try:
            self.advanced_window_ref.lift()
            self.advanced_window_ref.focus_force()
        except tk.TclError:
            pass
        return

    adv_window = tk.Toplevel(self)
    adv_window.title("Advanced Configuration & Optimization")
    adv_window.geometry("820x920")
    adv_window.minsize(680, 600)
    adv_window.transient(self)
    self.advanced_window_ref = adv_window

    def attach_tooltip(widget, text):
        if text:
            widget._hover_tooltip = HoverTooltip(widget, text)

    # Make UI scrollable in case vertical space is constrained
    container = ttk.Frame(adv_window)
    container.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(container, borderwidth=0, highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas, padding="15")

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas_frame_id = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

    def configure_canvas_width(event):
        canvas.itemconfig(canvas_frame_id, width=event.width)

    canvas.bind("<Configure>", configure_canvas_width)
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Enable mouse wheel scrolling in configuration window
    def _on_mousewheel(event):
        if event.num == 4:
            canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            canvas.yview_scroll(1, "units")
        elif event.delta:
            canvas.yview_scroll(int(-event.delta / 120), "units")

    def bind_tree_mousewheel(widget):
        widget.bind("<MouseWheel>", _on_mousewheel, add="+")
        widget.bind("<Button-4>", _on_mousewheel, add="+")
        widget.bind("<Button-5>", _on_mousewheel, add="+")
        for child in widget.winfo_children():
            bind_tree_mousewheel(child)

    adv_window.bind("<MouseWheel>", _on_mousewheel)
    adv_window.bind("<Button-4>", _on_mousewheel)
    adv_window.bind("<Button-5>", _on_mousewheel)

    # --- Live Hardware Memory Info Header ---
    hardware_info_frame = ttk.LabelFrame(scrollable_frame, text="Live Hardware & VRAM Diagnostics", padding="10")
    hardware_info_frame.pack(fill=tk.X, pady=(0, 5))
    hardware_info_frame.columnconfigure(1, weight=1)

    system_ram_lbl = ttk.Label(hardware_info_frame, text="System RAM:")
    system_ram_lbl.grid(row=0, column=0, sticky="w")
    system_ram_val_lbl = ttk.Label(hardware_info_frame, text="Detecting...", font=("Courier", 9))
    system_ram_val_lbl.grid(row=0, column=1, sticky="w", padx=10)

    gpu_vram_lbl = ttk.Label(hardware_info_frame, text="Total GPU VRAM:")
    gpu_vram_lbl.grid(row=1, column=0, sticky="w")
    gpu_vram_val_lbl = ttk.Label(hardware_info_frame, text="Detecting...", font=("Courier", 9))
    gpu_vram_val_lbl.grid(row=1, column=1, sticky="w", padx=10)

    app_vram_lbl = ttk.Label(hardware_info_frame, text="Total App VRAM:")
    app_vram_lbl.grid(row=2, column=0, sticky="w")
    app_vram_val_lbl = ttk.Label(hardware_info_frame, text="Detecting...", font=("Courier", 9, "bold"))
    app_vram_val_lbl.grid(row=2, column=1, sticky="w", padx=10)

    alloc_vram_lbl = ttk.Label(hardware_info_frame, text="  ↳ PyTorch Alloc:")
    alloc_vram_lbl.grid(row=3, column=0, sticky="w")
    alloc_vram_val_lbl = ttk.Label(hardware_info_frame, text="Detecting...", font=("Courier", 9))
    alloc_vram_val_lbl.grid(row=3, column=1, sticky="w", padx=10)

    cached_vram_lbl = ttk.Label(hardware_info_frame, text="  ↳ PyTorch Cache:")
    cached_vram_lbl.grid(row=4, column=0, sticky="w")
    cached_vram_val_lbl = ttk.Label(hardware_info_frame, text="Detecting...", font=("Courier", 9))
    cached_vram_val_lbl.grid(row=4, column=1, sticky="w", padx=10)

    def refresh_vram_diagnostics():
        if not adv_window.winfo_exists():
            return

        sys_stats = get_system_memory_stats_mb()
        sys_used = sys_stats.get('used_mb', 0)
        sys_total = sys_stats.get('total_mb', 0)
        if sys_total > 0:
            system_ram_val_lbl.config(text=f"{sys_used:.0f} MB / {sys_total:.0f} MB ({(sys_used/sys_total)*100:.1f}% used)")
        else:
            system_ram_val_lbl.config(text="Unknown / Not Available")

        total_gpu = get_gpu_vram_mb()
        if total_gpu > 0:
            gpu_vram_val_lbl.config(text=f"{total_gpu:.0f} MB")
        else:
            gpu_vram_val_lbl.config(text="No CUDA GPU found / Unknown")

        cuda_stats = get_cuda_memory_stats_mb()
        used_pt = cuda_stats.get('process_allocated_mb', 0)
        alloc_pt = cuda_stats.get('process_reserved_mb', 0)
        app_used = cuda_stats.get('app_used_mb', alloc_pt)
        max_pt = cuda_stats.get('max_allocated_mb', used_pt)
        
        if total_gpu > 0:
            app_vram_val_lbl.config(text=f"{app_used:.1f} MB (Includes PaddleOCR/Non-PyTorch)")
            alloc_vram_val_lbl.config(text=f"{used_pt:.1f} MB (Peak: {max_pt:.1f} MB)")
            cached_vram_val_lbl.config(text=f"{alloc_pt - used_pt:.1f} MB (Total Reserved: {alloc_pt:.1f} MB)")
        else:
            app_vram_val_lbl.config(text="N/A (CPU Mode)")
            alloc_vram_val_lbl.config(text="N/A (CPU Mode)")
            cached_vram_val_lbl.config(text="N/A (CPU Mode)")

        # Schedule next update in 1 second
        adv_window.after(1000, refresh_vram_diagnostics)

    refresh_vram_diagnostics()

    # --- Section: Concurrent Processing & Instance Orchestration ---
    concurrent_frame = ttk.LabelFrame(scrollable_frame, text="Concurrency & Pipeline Tuning", padding="10")
    concurrent_frame.pack(fill=tk.X, pady=5)
    concurrent_frame.columnconfigure(1, weight=1)

    def create_slider(parent, label_text, variable, from_value, to_value, tooltip_text):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=5)

        label = ttk.Label(row, text=label_text, width=28)
        label.pack(side=tk.LEFT)

        slider = ttk.Scale(row, from_=from_value, to=to_value, orient="horizontal")
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 10))

        value_var = tk.StringVar()
        value_label = ttk.Label(row, textvariable=value_var, width=5, anchor="e")
        value_label.pack(side=tk.LEFT)

        def refresh_value(*_):
            value_var.set(str(int(float(variable.get()))))

        def on_slider_change(value):
            variable.set(int(float(value)))
            refresh_value()

        slider.configure(variable=variable, command=on_slider_change)
        refresh_value()

        attach_tooltip(label, tooltip_text)
        attach_tooltip(slider, tooltip_text)
        return row

    profile_row = ttk.Frame(concurrent_frame)
    profile_row.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 8))
    ttk.Label(profile_row, text="Instance VRAM Profile:").pack(side=tk.LEFT, padx=(0, 8))

    pipeline_profile_combo = ttk.Combobox(
        profile_row,
        textvariable=self.pipeline_worker_profile,
        values=['Safe (Low VRAM)', 'Balanced (Medium VRAM)', 'Performance (High VRAM)', 'Custom'],
        state="readonly",
        width=25,
    )
    pipeline_profile_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
    attach_tooltip(pipeline_profile_combo, "Configures colorizer inference parameters for the selected profile.")

    # Frame to wrap the custom model/VRAM sliders
    custom_limits_frame = ttk.Frame(concurrent_frame)
    custom_limits_frame.grid(row=1, column=0, columnspan=3, sticky="ew")

    vram_breakdown_lbl = ttk.Label(concurrent_frame, text="", style="Italic.TLabel", justify="left")
    vram_breakdown_lbl.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 4))

    vram_warn_lbl = ttk.Label(concurrent_frame, text="", font=("Segoe UI", 9, "bold"), foreground="#d9534f", justify="left")
    vram_warn_lbl.grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 4))

    instance_cap_slider_row = create_slider(
        custom_limits_frame,
        "Colorizer Instance Cap",
        self.pipeline_colorize_instances,
        1,
        6,
        "Limits the number of concurrent model execution instances. More instances speed up batch processing but require substantial VRAM.",
    )

    width_slider_row = create_slider(
        custom_limits_frame,
        "Colorized Image Size",
        self.colorized_image_size,
        384,
        1536,
        "Target resolution for the model. Higher yields sharper colors but uses significantly more VRAM and is slower.",
    )
    width_slider_row.pack(fill=tk.X, pady=5)  # Always visible regardless of preset

    denoise_slider_row = create_slider(
        custom_limits_frame,
        "Denoiser Instance Cap",
        self.pipeline_denoise_instances,
        1,
        8,
        "Number of concurrent denoiser instances. More instances speed up batch denoising but use more VRAM.",
    )

    upscale_slider_row = create_slider(
        custom_limits_frame,
        "Upscaler Instance Cap",
        self.pipeline_upscale_instances,
        1,
        4,
        "Number of concurrent upscaler instances. Upscalers use significant VRAM per instance.",
    )

    writer_slider_row = create_slider(
        custom_limits_frame,
        "Writer Threads",
        self.pipeline_writer_threads,
        1,
        8,
        "Number of threads writing output images to disk. More threads reduce I/O bottlenecks on fast storage.",
    )

    adv_quality_trace_handles = []
    adv_ui_trace_handles = []

    def refresh_pipeline_runtime_hints(*_):
        if not adv_window.winfo_exists():
            return

        self._update_config_from_gui()

        total_gpu_mb = get_gpu_vram_mb()

        inst = self._active_pipeline_instances(device_name=None)
        prof = self._pipeline_instance_vram_profile_mb(device_name=None)
        
        breakdown = self._effective_pipeline_vram_breakdown_mb(device_name=None)

        # Update VRAM breakdown text
        breakdown_text = (
            f"Estimated VRAM footprint per colorizer instance: {prof.get('colorize', 0):.0f} MB\n"
            f"Inference concurrency: {inst.get('colorize', 0)} instance(s) × {THREADS_PER_INSTANCE.get('colorize', 1)} CPU thread(s)"
        )
        if self.enable_denoise.get():
            breakdown_text += f"\nDenoiser: {inst.get('denoise', 0)} instance(s) × {THREADS_PER_INSTANCE.get('denoise', 1)} CPU thread(s) ({prof.get('denoise', 0):.0f} MB/inst)"
        if self.enable_upscale.get():
            breakdown_text += f"\nUpscaler concurrency: {inst.get('upscale', 0)} instance(s) × {THREADS_PER_INSTANCE.get('upscale', 1)} CPU thread(s) ({prof.get('upscale', 0):.0f} MB/inst)"

        vram_breakdown_lbl.config(text=breakdown_text)

        # Warn if settings exceed total VRAM
        warning_msg = vram_warning_text(
            inst['denoise'],
            inst['colorize'],
            inst['upscale'],
            instance_profile_mb=prof,
            overhead_mb=self._pipeline_runtime_overhead_mb()
        )
        if warning_msg:
            vram_warn_lbl.config(text=warning_msg)
        else:
            vram_warn_lbl.config(text="")

    self._advanced_pipeline_runtime_refresher = refresh_pipeline_runtime_hints

    def on_pipeline_profile_combo_changed(*_):
        prof = self.pipeline_worker_profile.get()
        if prof == 'Custom':
            instance_cap_slider_row.pack(fill=tk.X, pady=5)
            denoise_slider_row.pack(fill=tk.X, pady=5)
            upscale_slider_row.pack(fill=tk.X, pady=5)
            writer_slider_row.pack(fill=tk.X, pady=5)
        else:
            instance_cap_slider_row.pack_forget()
            denoise_slider_row.pack_forget()
            upscale_slider_row.pack_forget()
            writer_slider_row.pack_forget()

            payload = INSTANCE_CAPS.get(prof)
            if payload:
                # Apply preset limits directly
                self.pipeline_colorize_instances.set(payload['max_instances'])
                self.colorized_size_to_set = payload['target_width']
                self.colorized_image_size.set(self.colorized_size_to_set)

        refresh_pipeline_runtime_hints()

    profile_trace = self.pipeline_worker_profile.trace_add('write', on_pipeline_profile_combo_changed)
    adv_ui_trace_handles.append((self.pipeline_worker_profile, profile_trace))
    on_pipeline_profile_combo_changed()  # Apply initial packaging state

    for live_var in (self.pipeline_colorize_instances, self.colorized_image_size, self.pipeline_denoise_instances, self.pipeline_upscale_instances, self.pipeline_writer_threads):
        trace_id = live_var.trace_add('write', lambda *_: refresh_pipeline_runtime_hints())
        adv_ui_trace_handles.append((live_var, trace_id))

    # --- Section: Model Paths ---
    paths_frame = ttk.LabelFrame(scrollable_frame, text="Model Paths", padding="10")
    paths_frame.pack(fill=tk.X, pady=5)
    paths_frame.columnconfigure(1, weight=1)

    ttk.Label(paths_frame, text="Colorizer Model:").grid(row=0, column=0, sticky="w", pady=2)
    colorizer_path_entry = ttk.Entry(paths_frame, textvariable=self.colorizer_path, width=50)
    colorizer_path_entry.grid(row=0, column=1, sticky="ew", padx=(8, 4), pady=2)
    def browse_colorizer():
        p = filedialog.askopenfilename(
            title="Select Colorizer Model",
            filetypes=(("ZIP Archives", "*.zip"), ("All files", "*.*")),
            parent=adv_window,
        )
        if p:
            self.colorizer_path.set(p)
    ttk.Button(paths_frame, text="Browse…", command=browse_colorizer).grid(row=0, column=2, padx=2, pady=2)

    # Upscaler Models Folder (NOT single file)
    ttk.Label(paths_frame, text="Upscaler Models Folder:").grid(row=1, column=0, sticky="w", pady=2)
    upscaler_dir_var = tk.StringVar(value=os.path.dirname(self.upscaler_path.get()))
    upscaler_dir_entry = ttk.Entry(paths_frame, textvariable=upscaler_dir_var, width=50)
    upscaler_dir_entry.grid(row=1, column=1, sticky="ew", padx=(8, 4), pady=2)
    def browse_upscaler_dir():
        p = filedialog.askdirectory(
            title="Select Upscaler Models Folder",
            initialdir=upscaler_dir_var.get(),
            parent=adv_window,
        )
        if p:
            upscaler_dir_var.set(p)
            refresh_model_list()
    ttk.Button(paths_frame, text="Browse…", command=browse_upscaler_dir).grid(row=1, column=2, padx=2, pady=2)
    attach_tooltip(upscaler_dir_entry, "Folder containing upscaler model files (.pth, .pt, .safetensors). Models are scanned and categorized automatically.")

    # --- Section: Model Selection (Categorized) ---
    import re as _re
    _MODEL_EXTENSIONS = {'.pth', '.pt', '.safetensors', '.onnx'}
    _CATEGORY_PATTERNS = {
        'Detail (Sharp Upscale)': [r'detail', r'V3detail'],
        'Denoise (Halftone Removal)': [r'denoise', r'V3denoise'],
        'BW Manga': [r'digimanga', r'_bw_', r'_bw\.'],
        'General Purpose': [r'RealESRGAN', r'AnimeSharp', r'UltraSharp', r'ESRGAN'],
    }
    # Ordered most-specific first — regex with word boundaries to prevent
    # 'DAT' matching inside 'FDAT', 'HAT' matching inside 'CHAT', etc.
    _ARCH_PATTERNS = [
        (r'(?:^|[_\-])FDAT[_\-]?XL(?:$|[_\-\.])', 'FDAT-XL'),
        (r'(?:^|[_\-])FDAT[_\-]?M(?:$|[_\-\.])',  'FDAT-M'),
        (r'(?:^|[_\-])FDAT(?:$|[_\-\.])',          'FDAT'),
        (r'(?:^|[_\-])DAT[_\-]?2(?:$|[_\-\.])',    'DAT-2'),
        (r'(?:^|[_\-])DAT(?:$|[_\-\.])',            'DAT'),
        (r'(?:^|[_\-])HAT[_\-]?L(?:$|[_\-\.])',    'HAT-L'),
        (r'(?:^|[_\-])HAT(?:$|[_\-\.])',            'HAT'),
        (r'(?:^|[_\-])SPAN[_\-]?S(?:$|[_\-\.])',   'SPAN-S'),
        (r'(?:^|[_\-])SPAN(?:$|[_\-\.])',           'SPAN'),
        (r'SPSR',                                    'SPSR'),
        (r'SRVGGNet',                                'SRVGGNet'),
        (r'GigaGAN|AuraSR',                         'GigaGAN'),
        (r'RealESRGAN|ESRGAN',                      'ESRGAN'),
    ]

    def _detect_arch(filename):
        name = os.path.splitext(filename)[0]
        for pattern, arch in _ARCH_PATTERNS:
            if _re.search(pattern, name, _re.IGNORECASE):
                return arch
        return 'Auto-Detect'
    _VRAM_TIERS = [
        (4096,  'SPAN-S, SRVGGNet', 'Lightweight — fastest, lowest VRAM'),
        (6144,  'DAT-2, FDAT-M',    'Balanced — good quality/speed tradeoff'),
        (8192,  'DAT-2, FDAT-M, HAT-L', 'High quality — needs 6-8GB free VRAM'),
        (99999, 'FDAT-XL, HAT-L',   'Best quality — 8GB+ recommended'),
    ]

    def _categorize_model(filename):
        name = os.path.splitext(filename)[0]
        for cat, patterns in _CATEGORY_PATTERNS.items():
            for pat in patterns:
                if _re.search(pat, name, _re.IGNORECASE):
                    return cat
        return 'Other'

    def _scan_models_folder(folder):
        """Scan folder for model files and categorize them."""
        categorized = {cat: [] for cat in _CATEGORY_PATTERNS}
        categorized['Other'] = []
        if not os.path.isdir(folder):
            return categorized
        try:
            for f in sorted(os.listdir(folder)):
                ext = os.path.splitext(f)[1].lower()
                if ext in _MODEL_EXTENSIONS:
                    cat = _categorize_model(f)
                    categorized[cat].append(f)
        except OSError:
            pass
        return categorized

    model_frame = ttk.LabelFrame(scrollable_frame, text="Upscaler Model Selection", padding="10")
    model_frame.pack(fill=tk.X, pady=5)
    model_frame.columnconfigure(1, weight=1)

    # VRAM recommendation
    vram_rec_var = tk.StringVar(value="")
    def _update_vram_rec():
        try:
            vram_mb = get_gpu_vram_mb()
            if vram_mb <= 0:
                vram_rec_var.set("CPU mode — use lightweight models (SPAN-S, SRVGGNet)")
                return
            for threshold, models, desc in _VRAM_TIERS:
                if vram_mb <= threshold:
                    vram_rec_var.set(f"GPU: {vram_mb/1024:.0f}GB → Recommended: {models} ({desc})")
                    return
        except Exception:
            vram_rec_var.set("")
    _update_vram_rec()

    vram_rec_lbl = ttk.Label(model_frame, textvariable=vram_rec_var, font=("Segoe UI", 8, "italic"), foreground="#5c8a2d")
    vram_rec_lbl.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 6))

    # Category selector
    category_var = tk.StringVar(value='All')
    ttk.Label(model_frame, text="Category:").grid(row=1, column=0, sticky="w", pady=3)
    category_combo = ttk.Combobox(
        model_frame, textvariable=category_var,
        values=['All', 'Detail (Sharp Upscale)', 'Denoise (Halftone Removal)', 'BW Manga', 'General Purpose', 'Other'],
        state="readonly", width=30
    )
    category_combo.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=3)
    attach_tooltip(category_combo, "Filter models by use case. Detail = clean art, Denoise = scanned manga with halftone, BW = pure B&W, General = all-purpose.")

    # Category description
    _CATEGORY_DESCS = {
        'All': 'Showing all models found in the folder.',
        'Detail (Sharp Upscale)': '🔍 Sharp lines, preserved detail. Best for clean digital source.',
        'Denoise (Halftone Removal)': '🧹 Removes halftone dots/noise while upscaling. Best for scanned manga.',
        'BW Manga': '⬛ Optimized for pure black & white manga ink lines.',
        'General Purpose': '🎨 General anime/photo upscaling. Good starting point.',
        'Other': 'Models that don\'t match known categories.',
    }
    cat_desc_var = tk.StringVar(value=_CATEGORY_DESCS.get('All', ''))
    ttk.Label(model_frame, textvariable=cat_desc_var, font=("Segoe UI", 8), foreground="#888").grid(
        row=2, column=0, columnspan=3, sticky="w", pady=(0, 4))

    # Model selector
    model_var = tk.StringVar(value=os.path.basename(self.upscaler_path.get()))
    ttk.Label(model_frame, text="Model:").grid(row=3, column=0, sticky="w", pady=3)
    model_combo = ttk.Combobox(model_frame, textvariable=model_var, state="readonly", width=45)
    model_combo.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=3)

    # Architecture (auto-detected, locked by default)
    arch_var = tk.StringVar(value=self.upscaler_type.get())
    arch_override_var = tk.BooleanVar(value=False)

    ttk.Label(model_frame, text="Architecture:").grid(row=4, column=0, sticky="w", pady=3)
    arch_combo = ttk.Combobox(
        model_frame, textvariable=arch_var,
        values=['Auto-Detect', 'DAT', 'DAT-2', 'HAT', 'HAT-L',
                'FDAT', 'FDAT-M', 'FDAT-XL', 'SPAN', 'SPAN-S',
                'ESRGAN', 'SRVGGNet', 'SPSR', 'GigaGAN'],
        state="disabled", width=20
    )
    arch_combo.grid(row=4, column=1, sticky="w", padx=(10, 0), pady=3)

    def _toggle_arch_override(*_):
        if arch_override_var.get():
            arch_combo.configure(state="readonly")
        else:
            # Revert to auto-detected value
            selected = model_var.get()
            if selected:
                detected = _detect_arch(selected)
                arch_var.set(detected)
                self.upscaler_type.set(detected)
            arch_combo.configure(state="disabled")

    arch_override_cb = ttk.Checkbutton(
        model_frame, text="Override", variable=arch_override_var,
        command=_toggle_arch_override
    )
    arch_override_cb.grid(row=4, column=2, sticky="w", padx=(4, 0), pady=3)
    attach_tooltip(arch_override_cb, "Unlock to manually set architecture. Only use if auto-detection from filename is wrong.")
    attach_tooltip(arch_combo, "Auto-detected from filename. Check 'Override' to change manually.")

    # Model info label (shows detected arch + file size)
    model_info_var = tk.StringVar(value="")
    ttk.Label(model_frame, textvariable=model_info_var, font=("Courier", 8), foreground="#888").grid(
        row=5, column=0, columnspan=3, sticky="w", pady=(2, 0))

    # --- Model list management ---
    _cached_models = {}  # category -> [filename]

    def refresh_model_list(*_):
        nonlocal _cached_models
        folder = upscaler_dir_var.get().strip()
        _cached_models = _scan_models_folder(folder)
        _filter_by_category()

    def _filter_by_category(*_):
        cat = category_var.get()
        cat_desc_var.set(_CATEGORY_DESCS.get(cat, ''))
        if cat == 'All':
            all_models = []
            for models in _cached_models.values():
                all_models.extend(models)
            model_combo['values'] = sorted(set(all_models))
        else:
            model_combo['values'] = _cached_models.get(cat, [])
        # Keep current selection if still valid
        current = model_var.get()
        vals = model_combo['values']
        if current not in vals and vals:
            model_var.set(vals[0])

    def _on_model_selected(*_):
        selected = model_var.get()
        folder = upscaler_dir_var.get().strip()
        if selected:
            full_path = os.path.join(folder, selected)
            self.upscaler_path.set(full_path)
            # Auto-detect architecture from filename
            detected = _detect_arch(selected)
            arch_var.set(detected)
            self.upscaler_type.set(detected)
            # Reset override — new model = fresh auto-detect
            arch_override_var.set(False)
            arch_combo.configure(state="disabled")
            # Show file info
            try:
                size_mb = os.path.getsize(full_path) / (1024 * 1024)
                model_info_var.set(f"📁 {full_path}  ({size_mb:.1f} MB)  Arch: {detected}")
            except OSError:
                model_info_var.set(f"📁 {full_path}")

    def _on_arch_changed(*_):
        self.upscaler_type.set(arch_var.get())

    category_combo.bind("<<ComboboxSelected>>", _filter_by_category)
    model_combo.bind("<<ComboboxSelected>>", _on_model_selected)
    arch_combo.bind("<<ComboboxSelected>>", _on_arch_changed)
    refresh_model_list()  # Initial scan

    # --- Section: Performance Sliders ---
    perf_frame = ttk.LabelFrame(scrollable_frame, text="Performance Tuning", padding="10")
    perf_frame.pack(fill=tk.X, pady=5)

    create_slider(perf_frame, "Upscaler Tile Size", self.upscaler_tile_size, 0, 1024,
                  "Tile size for upscaler inference. 0 = process full image (more VRAM). 128-512 recommended for large images.")
    create_slider(perf_frame, "Colorizer Tile Size", self.colorizer_tile_size, 0, 1024,
                  "Tile size for colorizer inference. 0 = full image (default). Only set if VRAM limited.")
    create_slider(perf_frame, "Tile Padding", self.tile_pad, 0, 64,
                  "Overlap padding between tiles to prevent seam artifacts. Default: 8")
    create_slider(perf_frame, "Input Image Size", self.input_image_size, 256, 4096,
                  "Max input dimension for processing. Higher values preserve detail but use more VRAM. New architectures (DAT, HAT, FDAT) handle larger inputs well.")
    create_slider(perf_frame, "Denoise Max Side", self.denoise_max_side, 0, 4096,
                  "Max dimension for denoiser. 0 = unlimited (use input size). Set lower to speed up denoising on very large images.")

    # --- Section: Precision & Acceleration ---
    engine_frame = ttk.LabelFrame(scrollable_frame, text="Precision & Acceleration", padding="10")
    engine_frame.pack(fill=tk.X, pady=5)
    engine_frame.columnconfigure(1, weight=1)

    # Hardware summary
    hw_summary_lbl = ttk.Label(engine_frame, textvariable=self.precision_hw_summary, font=("Courier", 8), foreground="#888")
    hw_summary_lbl.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))

    precision_options = ['Auto (BF16 > FP16 > FP32)', 'BF16', 'FP16', 'FP32']
    if supports_fp8():
        precision_options.append('FP8 (experimental)')

    ttk.Label(engine_frame, text="Colorize Precision:").grid(row=1, column=0, sticky="w", pady=3)
    colorize_prec_combo = ttk.Combobox(
        engine_frame, textvariable=self.colorize_precision_policy,
        values=precision_options, state="readonly", width=30
    )
    colorize_prec_combo.grid(row=1, column=1, sticky="ew", padx=(10, 0), pady=3)
    attach_tooltip(colorize_prec_combo, "Precision for colorizer inference. BF16 recommended for RTX 30/40/50 series.")

    ttk.Label(engine_frame, text="Upscale Precision:").grid(row=2, column=0, sticky="w", pady=3)
    upscale_prec_combo = ttk.Combobox(
        engine_frame, textvariable=self.upscale_precision_policy,
        values=precision_options, state="readonly", width=30
    )
    upscale_prec_combo.grid(row=2, column=1, sticky="ew", padx=(10, 0), pady=3)
    attach_tooltip(upscale_prec_combo, "Precision for upscaler inference. DAT2/HAT-L models are BF16 natively, FDAT-M is FP16.")

    ttk.Checkbutton(engine_frame, text="Cast model weights to match precision", variable=self.precision_cast_weights).grid(
        row=3, column=0, columnspan=2, sticky="w", pady=2)

    ttk.Label(engine_frame, text="Fallback Mode:").grid(row=4, column=0, sticky="w", pady=3)
    fallback_combo = ttk.Combobox(
        engine_frame, textvariable=self.precision_fallback,
        values=['Per-image FP32 retry', 'Disable AMP after first error', 'Abort on precision error'],
        state="readonly", width=30
    )
    fallback_combo.grid(row=4, column=1, sticky="ew", padx=(10, 0), pady=3)
    attach_tooltip(fallback_combo, "What to do when a precision error occurs during inference.")

    ttk.Checkbutton(engine_frame, text="Allow TF32 (faster matmul on Ampere+)", variable=self.precision_allow_tf32).grid(
        row=5, column=0, columnspan=2, sticky="w", pady=2)

    # --- Section: Inference Optimizations (unified, no duplicates) ---
    opt_frame = ttk.LabelFrame(scrollable_frame, text="Inference Optimizations", padding="10")
    opt_frame.pack(fill=tk.X, pady=5)

    ttk.Checkbutton(
        opt_frame, text="Compile inference graphs (PyTorch 2.x torch.compile)",
        variable=self.torch_compile
    ).pack(anchor="w", pady=2)
    attach_tooltip(opt_frame.winfo_children()[-1], "JIT-compiles model graphs for faster execution. First run is slower (compilation).")

    flash_cb = ttk.Checkbutton(
        opt_frame, text="Enable FlashAttention / SDPA",
        variable=self.enable_flash_attention
    )
    flash_cb.pack(anchor="w", pady=2)
    _flash_cb_original_text = "Enable FlashAttention / SDPA"
    attach_tooltip(flash_cb, "Uses PyTorch's built-in efficient attention kernels (FlashAttention2, memory-efficient). Superseded when SageAttention is active.")

    ttk.Checkbutton(
        opt_frame, text="Enable CPU Offload (lower VRAM, slower)",
        variable=self.enable_cpu_offload
    ).pack(anchor="w", pady=2)

    # SageAttention sub-section
    sage_sep = ttk.Separator(opt_frame, orient="horizontal")
    sage_sep.pack(fill=tk.X, pady=(8, 4))
    ttk.Label(opt_frame, text="SageAttention (RTX 30/40/50)", font=("Segoe UI", 9, "bold")).pack(anchor="w")

    sage_status_var = tk.StringVar(value="Checking...")
    sage_status_lbl = ttk.Label(opt_frame, textvariable=sage_status_var, font=("Courier", 8), foreground="#888")
    sage_status_lbl.pack(anchor="w", pady=(2, 4))

    sage_cb = ttk.Checkbutton(
        opt_frame, text="Enable SageAttention (~2-3× faster attention)",
        variable=self.enable_sage_attention
    )
    sage_cb.pack(anchor="w", pady=2)
    attach_tooltip(sage_cb,
                   "Replaces PyTorch's entire SDPA with SageAttention's faster CUDA kernel. "
                   "Supersedes FlashAttention (both use the same hook point). "
                   "Benefits DAT, DAT-2, HAT, HAT-L, and FDAT upscaler models.")

    sage_install_btn = ttk.Button(opt_frame, text="Install SageAttention")
    sage_install_btn.pack(anchor="w", pady=(4, 2))

    # --- SageAttention / FlashAttention interlock ---
    def _update_sage_flash_interlock(*_):
        """SageAttention supersedes FlashAttention (replaces F.scaled_dot_product_attention).
        When SA is on, FA is irrelevant — grey it out to avoid confusion."""
        if self.enable_sage_attention.get():
            flash_cb.configure(state="disabled", text="FlashAttention / SDPA (superseded by SageAttention)")
        else:
            flash_cb.configure(state="normal", text=_flash_cb_original_text)

    self.enable_sage_attention.trace_add('write', _update_sage_flash_interlock)
    adv_ui_trace_handles.append((self.enable_sage_attention,
                                  self.enable_sage_attention.trace_info()[-1][1]))
    _update_sage_flash_interlock()  # Apply initial state

    def refresh_sage_status():
        """Update status label and grey out Install button if already installed."""
        try:
            from Backend.sage_attention import detect_compatibility, get_installed_version
            compat = detect_compatibility()
            version = get_installed_version()
            if version:
                sage_status_var.set(f"✓ Installed: v{version} | {compat['message']}")
                sage_status_lbl.config(foreground="#2d8a4e")
                sage_install_btn.configure(state="disabled", text="✓ SageAttention Installed")
                sage_cb.configure(state="normal")
            else:
                sage_status_var.set(f"✗ Not installed | {compat['message']}")
                sage_status_lbl.config(foreground="#d9534f")
                sage_install_btn.configure(state="normal", text="Install SageAttention")
                # Disable the enable checkbox if not installed
                sage_cb.configure(state="disabled")
                if self.enable_sage_attention.get():
                    self.enable_sage_attention.set(False)
        except Exception as e:
            sage_status_var.set(f"Error: {e}")

    def install_sage_action():
        sage_status_var.set("Installing... (this may take 30-60 seconds)")
        sage_status_lbl.config(foreground="#f0ad4e")
        sage_install_btn.configure(state="disabled", text="Installing...")
        adv_window.update_idletasks()

        import threading
        def _install():
            try:
                from Backend.sage_attention import install_sageattention
                ok, msg = install_sageattention(log_callback=lambda m: print(m))
                if adv_window.winfo_exists():
                    adv_window.after(0, lambda: sage_status_var.set(f"{'✓' if ok else '✗'} {msg}"))
                    adv_window.after(0, lambda: sage_status_lbl.config(foreground="#2d8a4e" if ok else "#d9534f"))
                    adv_window.after(100, refresh_sage_status)
            except Exception as e:
                if adv_window.winfo_exists():
                    adv_window.after(0, lambda: sage_status_var.set(f"Error: {e}"))
                    adv_window.after(0, lambda: sage_install_btn.configure(state="normal", text="Retry Install"))
        threading.Thread(target=_install, daemon=True).start()

    sage_install_btn.configure(command=install_sage_action)
    refresh_sage_status()

    # --- Section: Colorizer Reliability ---
    reliability_frame = ttk.LabelFrame(scrollable_frame, text="Colorizer Reliability", padding="10")
    reliability_frame.pack(fill=tk.X, pady=5)

    ttk.Checkbutton(
        reliability_frame,
        text="Lock colorizer internal width to 576px (force safe mode)",
        variable=self.force_safe_colorizer_width
    ).pack(anchor="w", pady=2)
    attach_tooltip(reliability_frame.winfo_children()[-1],
                   "Forces the colorizer to use 576px internal width regardless of Colorized Image Size slider. "
                   "Enable if you see artifacts or crashes with higher resolutions.")

    # --- Section: Logging & Diagnostics ---
    log_frame = ttk.LabelFrame(scrollable_frame, text="Logging & Diagnostics", padding="10")
    log_frame.pack(fill=tk.X, pady=5)

    ttk.Checkbutton(log_frame, text="Enable detailed per-image debug logs", variable=self.detailed_debug_logs).pack(anchor="w", pady=2)
    ttk.Checkbutton(log_frame, text="Export OCR diagnostic overlays", variable=self.export_ocr_debug).pack(anchor="w", pady=2)

    # --- Section: OCR & Bubble Settings ---
    ocr_frame = ttk.LabelFrame(scrollable_frame, text="OCR & Bubble Settings", padding="10")
    ocr_frame.pack(fill=tk.X, pady=5)

    ttk.Label(ocr_frame, text="OCR Model Tier:").pack(anchor="w")
    ocr_tier_combo = ttk.Combobox(ocr_frame, textvariable=self.ocr_model_tier,
                                   values=['server', 'mobile'], state="readonly")
    ocr_tier_combo.pack(fill=tk.X, pady=(2, 8))
    attach_tooltip(ocr_tier_combo, "'server' = PP-OCRv5 server (more accurate, slower). 'mobile' = PP-OCRv5 mobile (faster, less accurate).")

    # Max OCR Dimension — use tk.Scale for resolution=256 support
    ocr_dim_row = ttk.Frame(ocr_frame)
    ocr_dim_row.pack(fill=tk.X, pady=4)
    ttk.Label(ocr_dim_row, text="Max OCR Dimension:", width=28).pack(side=tk.LEFT)
    ocr_dim_val_var = tk.StringVar(value=str(self.max_ocr_dimension.get()))
    ocr_dim_scale = tk.Scale(ocr_dim_row, from_=2048, to=5120, resolution=256,
                              orient=tk.HORIZONTAL, variable=self.max_ocr_dimension,
                              showvalue=False)
    ocr_dim_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 10))
    ocr_dim_lbl = ttk.Label(ocr_dim_row, textvariable=ocr_dim_val_var, width=5, anchor="e")
    ocr_dim_lbl.pack(side=tk.LEFT)
    def _update_ocr_dim_label(*_):
        ocr_dim_val_var.set(str(self.max_ocr_dimension.get()))
    self.max_ocr_dimension.trace_add('write', _update_ocr_dim_label)
    attach_tooltip(ocr_dim_scale, "Maximum image dimension for OCR detection. Higher catches more text but uses more VRAM. Range: 2048-5120, step 256.")

    ttk.Label(ocr_frame, text="YOLO Bubbles:").pack(anchor="w")
    yolo_combo = ttk.Combobox(ocr_frame, textvariable=self.use_yolo_bubbles,
                               values=['full', 'yolo_only', 'off'], state="readonly")
    yolo_combo.pack(fill=tk.X, pady=(2, 8))
    attach_tooltip(yolo_combo, "'full' = YOLO + OCR detection. 'yolo_only' = YOLO bubbles only (no text detection). 'off' = disable bubble detection.")

    ttk.Label(ocr_frame, text="YOLO Model Type:").pack(anchor="w")
    yolo_model_combo = ttk.Combobox(ocr_frame, textvariable=self.yolo_model_type,
                                     values=['seg', 'det', 'both'], state="readonly")
    yolo_model_combo.pack(fill=tk.X, pady=(2, 8))
    attach_tooltip(yolo_model_combo, "'seg' = segmentation model (pixel masks). 'det' = detection model (bounding boxes). 'both' = merge results from both.")

    # FP Strictness — use tk.Scale for float resolution=0.05
    fp_row = ttk.Frame(ocr_frame)
    fp_row.pack(fill=tk.X, pady=4)
    ttk.Label(fp_row, text="FP Strictness:", width=28).pack(side=tk.LEFT)
    fp_val_var = tk.StringVar(value=f"{self.fp_strictness.get():.2f}")
    fp_scale = tk.Scale(fp_row, from_=0.0, to=1.0, resolution=0.05,
                         orient=tk.HORIZONTAL, variable=self.fp_strictness,
                         showvalue=False)
    fp_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 10))
    fp_lbl = ttk.Label(fp_row, textvariable=fp_val_var, width=5, anchor="e")
    fp_lbl.pack(side=tk.LEFT)
    def _update_fp_label(*_):
        fp_val_var.set(f"{self.fp_strictness.get():.2f}")
    self.fp_strictness.trace_add('write', _update_fp_label)
    attach_tooltip(fp_scale, "False positive rejection strictness. 0.0 = permissive (keep more bubbles), 1.0 = aggressive (reject more). Default: 0.5")

    # SFX Feather — use tk.Scale for resolution=1, correct range 0-10
    sfx_row = ttk.Frame(ocr_frame)
    sfx_row.pack(fill=tk.X, pady=4)
    ttk.Label(sfx_row, text="SFX Feather Radius:", width=28).pack(side=tk.LEFT)
    sfx_val_var = tk.StringVar(value=str(self.sfx_feather_radius.get()))
    sfx_scale = tk.Scale(sfx_row, from_=0, to=10, resolution=1,
                          orient=tk.HORIZONTAL, variable=self.sfx_feather_radius,
                          showvalue=False)
    sfx_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 10))
    sfx_lbl = ttk.Label(sfx_row, textvariable=sfx_val_var, width=5, anchor="e")
    sfx_lbl.pack(side=tk.LEFT)
    def _update_sfx_label(*_):
        sfx_val_var.set(str(self.sfx_feather_radius.get()))
    self.sfx_feather_radius.trace_add('write', _update_sfx_label)
    attach_tooltip(sfx_scale, "Feathering radius for SFX (sound effects) mask edges. 0 = no feather, 10 = max softness.")

    # (Upscaler Architecture Override is now integrated into the Model Selection section above)

    # --- Section: Denoise Sigma ---
    denoise_frame = ttk.LabelFrame(scrollable_frame, text="Denoise Sigma", padding="10")
    denoise_frame.pack(fill=tk.X, pady=5)
    create_slider(denoise_frame, "Value", self.denoise_sigma, 1, 100, "Strength of the denoiser. Default: 25")
    preset_frame = ttk.LabelFrame(scrollable_frame, text="Folder-Specific & Global Presets", padding="10")
    preset_frame.pack(fill=tk.X, pady=5)
    preset_frame.columnconfigure(1, weight=1)

    ttk.Label(preset_frame, text="Active Preset:").grid(row=0, column=0, sticky="w")
    preset_name_var = tk.StringVar()
    preset_selector = ttk.Combobox(preset_frame, textvariable=preset_name_var, state="readonly")
    preset_selector.grid(row=0, column=1, columnspan=2, sticky="ew", padx=(10, 0), pady=5)

    def refresh_adv_preset_selector(selected_name=None):
        values = self._ordered_preset_names()
        preset_selector['values'] = values

        folder_path = self.input_folder.get().strip().strip("'\"")
        fallback = self._active_quality_preset_for_folder(folder_path)

        if selected_name in self.quality_presets:
            preset_name_var.set(selected_name)
        elif fallback in self.quality_presets:
            preset_name_var.set(fallback)
        else:
            preset_name_var.set('Default')

    self._advanced_preset_selector_refresher = refresh_adv_preset_selector

    # Preset Action Buttons Grid
    actions_grid_frame = ttk.Frame(preset_frame)
    actions_grid_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(5, 0))
    actions_grid_frame.columnconfigure((0, 1, 2), weight=1)

    set_folder_button = ttk.Button(actions_grid_frame, text="Apply Folder")
    set_folder_button.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
    attach_tooltip(set_folder_button, "Binds the chosen preset to the selected input directory.")

    set_global_button = ttk.Button(actions_grid_frame, text="Apply Global")
    set_global_button.grid(row=0, column=1, padx=2, pady=2, sticky="ew")
    attach_tooltip(set_global_button, "Sets the chosen preset as default for directories without specific overrides.")

    revert_default_button = ttk.Button(actions_grid_frame, text="Clear Override")
    revert_default_button.grid(row=0, column=2, padx=2, pady=2, sticky="ew")
    attach_tooltip(revert_default_button, "Reverts active preset to Default.")

    save_preset_button = ttk.Button(actions_grid_frame, text="Save Preset")
    save_preset_button.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
    attach_tooltip(save_preset_button, "Creates a new user preset using the current quality settings.")

    rename_preset_button = ttk.Button(actions_grid_frame, text="Rename Selected")
    rename_preset_button.grid(row=1, column=1, padx=2, pady=2, sticky="ew")
    attach_tooltip(rename_preset_button, "Renames the selected custom preset.")

    delete_preset_button = ttk.Button(actions_grid_frame, text="Delete Selected")
    delete_preset_button.grid(row=1, column=2, padx=2, pady=2, sticky="ew")
    attach_tooltip(delete_preset_button, "Deletes the selected custom preset.")

    import_preset_button = ttk.Button(actions_grid_frame, text="Import Presets…")
    import_preset_button.grid(row=2, column=0, padx=2, pady=2, sticky="ew")
    attach_tooltip(import_preset_button, "Imports quality presets from a JSON file.")

    export_preset_button = ttk.Button(actions_grid_frame, text="Export Presets…")
    export_preset_button.grid(row=2, column=1, padx=2, pady=2, sticky="ew")
    attach_tooltip(export_preset_button, "Exports all presets to a JSON file.")

    def selected_preset_name():
        return preset_name_var.get().strip()

    def sync_active_preset_from_quality_vars(*_):
        # Scan if active GUI variables match any defined preset
        current_gui_dict = self._snapshot_quality_preset_from_vars()
        matching_preset = 'Custom'

        for name, payload in self.quality_presets.items():
            payload_coerced = self._coerce_quality_preset_payload(payload)
            match = True
            for k, v in payload_coerced.items():
                if current_gui_dict.get(k) != v:
                    match = False
                    break
            if match:
                matching_preset = name
                break

        preset_name_var.set(matching_preset)

    def on_adv_preset_selected(event=None):
        selected = selected_preset_name()
        if selected not in self.quality_presets:
            return

        self._apply_quality_preset_payload_to_vars(self.quality_presets[selected])
        self._sync_open_preset_selectors(source='advanced', selected_name=selected)

    def set_active(scope):
        selected = selected_preset_name()
        if selected not in self.quality_presets:
            messagebox.showerror("Preset", f"Preset '{selected}' does not exist.", parent=adv_window)
            return

        folder_path = self.input_folder.get().strip().strip("'\"")
        if not self._set_active_quality_preset(selected, folder_path=folder_path, scope=scope):
            messagebox.showerror("Preset", "Failed to bind preset as active.", parent=adv_window)
            return

        self.save_settings(snapshot_active_preset=False)
        self._sync_open_preset_selectors(source='advanced', selected_name=selected)
        messagebox.showinfo("Preset", f"Preset '{selected}' is now active ({scope}).", parent=adv_window)

    def save_preset_action():
        new_name = simpledialog.askstring("New Preset", "Preset name:", parent=adv_window)
        if new_name is None:
            return

        clean_name = new_name.strip()
        if not clean_name:
            messagebox.showerror("Preset", "Name cannot be empty.", parent=adv_window)
            return

        allow_overwrite = False
        if clean_name in self.quality_presets:
            if clean_name == 'Default':
                messagebox.showinfo("Preset", "Use a different name; Default preset cannot be modified.", parent=adv_window)
                return
            if not messagebox.askyesno("Overwrite Preset", f"Preset '{clean_name}' exists. Overwrite?", parent=adv_window):
                return
            allow_overwrite = True

        ok, result = self._save_current_values_as_preset(clean_name, allow_overwrite=allow_overwrite)
        if not ok:
            messagebox.showerror("Preset", result, parent=adv_window)
            return

        folder_path = self.input_folder.get().strip().strip("'\"")
        if folder_path and os.path.isdir(folder_path):
            self._set_active_quality_preset(result, folder_path=folder_path, scope='folder')
        else:
            self._set_active_quality_preset(result, scope='global')
        self.save_settings()
        refresh_adv_preset_selector(result)
        self._sync_open_preset_selectors(source='advanced')

    def rename_preset_action():
        selected = selected_preset_name()
        new_name = simpledialog.askstring("Rename Preset", "New preset name:", initialvalue=selected, parent=adv_window)
        if new_name is None:
            return
        ok, result = self._rename_quality_preset(selected, new_name)
        if not ok:
            messagebox.showerror("Preset", result, parent=adv_window)
            return
        self.save_settings()
        refresh_adv_preset_selector(result)
        self._sync_open_preset_selectors(source='advanced')

    def delete_preset_action():
        selected = selected_preset_name()
        if not messagebox.askyesno("Delete Preset", f"Delete preset '{selected}'?", parent=adv_window):
            return
        ok, result = self._delete_quality_preset(selected)
        if not ok:
            messagebox.showerror("Preset", result, parent=adv_window)
            return
        self.save_settings()
        refresh_adv_preset_selector(self._active_quality_preset_for_folder(self.input_folder.get().strip().strip("'\"")))
        self._sync_open_preset_selectors(source='advanced')

    def import_preset_action():
        import_path = filedialog.askopenfilename(
            title="Import Presets",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
            parent=adv_window,
        )
        if not import_path:
            return
        try:
            merged_count = self._import_quality_presets(import_path)
        except Exception as err:
            messagebox.showerror("Preset Import", f"Failed to import presets: {err}", parent=adv_window)
            return

        self.save_settings()
        refresh_adv_preset_selector()
        self._sync_open_preset_selectors(source='advanced')
        messagebox.showinfo("Preset Import", f"Imported {merged_count} preset(s).", parent=adv_window)

    def export_preset_action():
        export_path = filedialog.asksaveasfilename(
            title="Export Presets",
            defaultextension=".json",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
            initialfile="colorizer-presets.json",
            parent=adv_window,
        )
        if not export_path:
            return
        try:
            self._export_quality_presets(export_path)
        except Exception as err:
            messagebox.showerror("Preset Export", f"Failed to export presets: {err}", parent=adv_window)
            return

        messagebox.showinfo("Preset Export", f"Presets exported to:\n{export_path}", parent=adv_window)

    set_folder_button.config(command=lambda: set_active('folder'))
    set_global_button.config(command=lambda: set_active('global'))
    revert_default_button.config(command=lambda: (preset_name_var.set('Default'), set_active('folder')))
    save_preset_button.config(command=save_preset_action)
    rename_preset_button.config(command=rename_preset_action)
    delete_preset_button.config(command=delete_preset_action)
    import_preset_button.config(command=import_preset_action)
    export_preset_button.config(command=export_preset_action)
    preset_selector.bind("<<ComboboxSelected>>", on_adv_preset_selected)
    refresh_adv_preset_selector()

    quality_frame = ttk.LabelFrame(scrollable_frame, text="Color Transfer Quality", padding="10")
    quality_frame.pack(fill=tk.X, pady=5)
    ttk.Label(quality_frame, text="Chroma Resize Mode:").pack(anchor="w")
    chroma_resize_combo = ttk.Combobox(
        quality_frame,
        textvariable=self.chroma_resize_mode,
        values=['BICUBIC', 'BILINEAR', 'LANCZOS'],
        state="readonly"
    )
    chroma_resize_combo.pack(fill=tk.X, pady=(2, 8))

    ttk.Checkbutton(
        quality_frame,
        text="Edge-aware chroma protection",
        variable=self.edge_chroma_protection
    ).pack(anchor="w")
    create_slider(
        quality_frame,
        "Edge Chroma Strength",
        self.edge_chroma_strength,
        0,
        100,
        "How aggressively color is reduced on strong line edges. Higher reduces halos but may mute very thin colors."
    ).pack(fill=tk.X)

    ttk.Checkbutton(
        quality_frame,
        text="Line/ink color bleed protection",
        variable=self.line_ink_protection
    ).pack(anchor="w")
    create_slider(
        quality_frame,
        "Line/Ink Strength",
        self.line_ink_protection_strength,
        0,
        100,
        "Extra chroma suppression on dark edge-like line-art regions."
    ).pack(fill=tk.X)

    ttk.Checkbutton(
        quality_frame,
        text="Screentone chroma smoothing",
        variable=self.screentone_chroma_smoothing
    ).pack(anchor="w")
    create_slider(
        quality_frame,
        "Screentone Smoothing Strength",
        self.screentone_smoothing_strength,
        0,
        100,
        "Smooths chroma in high-frequency screentone regions to reduce color speckle."
    ).pack(fill=tk.X)



    for live_var in (
        self.denoise_sigma,
        self.chroma_resize_mode,
        self.edge_chroma_protection,
        self.edge_chroma_strength,
        self.line_ink_protection,
        self.line_ink_protection_strength,
        self.screentone_chroma_smoothing,
        self.screentone_smoothing_strength,
    ):
        trace_id = live_var.trace_add('write', sync_active_preset_from_quality_vars)
        adv_quality_trace_handles.append((live_var, trace_id))

    def cleanup_adv_preset_sync(event=None):
        if event is not None and event.widget is not adv_window:
            return

        if self._advanced_preset_selector_refresher == refresh_adv_preset_selector:
            self._advanced_preset_selector_refresher = None

        if self._advanced_pipeline_runtime_refresher == refresh_pipeline_runtime_hints:
            self._advanced_pipeline_runtime_refresher = None

        for variable, trace_id in adv_quality_trace_handles:
            try:
                variable.trace_remove('write', trace_id)
            except tk.TclError:
                pass

        for variable, trace_id in adv_ui_trace_handles:
            try:
                variable.trace_remove('write', trace_id)
            except tk.TclError:
                pass

    adv_window.bind("<Destroy>", cleanup_adv_preset_sync, add="+")

    ttk.Button(scrollable_frame, text="Close", command=adv_window.destroy).pack(side=tk.BOTTOM, pady=10)

# ui/preview_window.py
import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, ttk, simpledialog, messagebox
import threading
from collections import OrderedDict, deque
import random

import PIL.Image
import PIL.ImageDraw
import numpy as np
import torch
from PIL import ImageTk

from ui.utils import HoverTooltip

def open_preview_window(self):
    from ui.main_window import (
        DEFAULT_INPUT_WIDTH,
        PREVIEW_SOURCE_CACHE_SIZE,
        PREVIEW_DENOISE_CACHE_SIZE,
        PREVIEW_RERUN_DEBOUNCE_MS,
        sanitize_input_width_limit,
        clamp_image_to_input_width,
        transfer_luminance_from_source,
        compute_transfer_debug_masks,
        apply_debug_mask_visual,
    )
    from precision_utils import supports_bf16
    from transfer_quality import (
        DEBUG_MASK_UI_VALUES,
        DEBUG_MASK_COLOR_MAP,
        debug_mask_label_to_key,
    )
    from mask_bundle_export import export_current_mask_bundle
    from pipeline import THREADS_PER_INSTANCE

    image_folder = self.input_folder.get().strip().strip("'\"")
    if not image_folder or not os.path.isdir(image_folder):
        self.log("[!] Please select a valid input folder first to use the preview.")
        return

    supported_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    image_paths = []
    for root, _, files in os.walk(image_folder):
        for filename in files:
            if filename.lower().endswith(supported_ext):
                image_paths.append(os.path.join(root, filename))

    image_paths.sort()
    if not image_paths:
        self.log(f"[!] No images found in '{os.path.basename(image_folder)}' for preview.")
        return

    if self.preview_session and self.preview_session.get('window'):
        prior_window = self.preview_session['window']
        try:
            if prior_window.winfo_exists():
                prior_window.destroy()
        except tk.TclError:
            pass

    active_folder_preset = self._active_quality_preset_for_folder(image_folder)
    if active_folder_preset in self.quality_presets:
        self._apply_quality_preset_payload_to_vars(self.quality_presets[active_folder_preset])

    preview_window = tk.Toplevel(self)
    preview_window.title("Colorizer Preview")
    preview_window.geometry("1320x900")
    preview_window.minsize(980, 700)
    preview_window.transient(self)

    def attach_tooltip(widget, text):
        if text:
            widget._hover_tooltip = HoverTooltip(widget, text)

    main_frame = ttk.Frame(preview_window, padding="6")
    main_frame.pack(fill=tk.BOTH, expand=True)
    main_frame.columnconfigure(0, weight=1)
    main_frame.rowconfigure(4, weight=1)  # image pane gets all stretch

    # --- Row 0: Navigation + Status (compact single row) ---
    nav_frame = ttk.Frame(main_frame)
    nav_frame.grid(row=0, column=0, sticky="ew", pady=(0, 2))
    nav_frame.columnconfigure(5, weight=1)

    prev_button = ttk.Button(nav_frame, text="◀ Prev", width=7)
    prev_button.grid(row=0, column=0, padx=(0, 2))
    next_button = ttk.Button(nav_frame, text="Next ▶", width=7)
    next_button.grid(row=0, column=1, padx=2)
    random_button = ttk.Button(nav_frame, text="🎲", width=3)
    random_button.grid(row=0, column=2, padx=2)
    rerun_button = ttk.Button(nav_frame, text="⟳ Re-run", width=8)
    rerun_button.grid(row=0, column=3, padx=(2, 6))

    image_info_var = tk.StringVar(value="")
    ttk.Label(nav_frame, textvariable=image_info_var,
              font=("Segoe UI", 9)).grid(row=0, column=5, sticky="w", padx=4)

    status_var = tk.StringVar(value="Ready")
    ttk.Label(nav_frame, textvariable=status_var,
              style="Italic.TLabel").grid(row=0, column=6, sticky="e", padx=4)

    # --- Row 1: Options row (zoom, wipe, fast mode) ---
    opts_frame = ttk.Frame(main_frame)
    opts_frame.grid(row=1, column=0, sticky="ew", pady=(0, 2))

    preview_bf16_supported = supports_bf16()
    if not preview_bf16_supported:
        self.preview_fast_mode.set(False)
    fast_mode_check = ttk.Checkbutton(opts_frame, text="BF16 Fast",
                                      variable=self.preview_fast_mode)
    fast_mode_check.pack(side=tk.LEFT, padx=(0, 6))
    if not preview_bf16_supported:
        fast_mode_check.state(["disabled"])
        attach_tooltip(fast_mode_check, "BF16 not supported on this GPU.")
    else:
        attach_tooltip(fast_mode_check, "Force BF16 for faster previews.")

    pixel_wipe_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts_frame, text="Pixel-Wipe",
                    variable=pixel_wipe_var).pack(side=tk.LEFT, padx=6)

    ttk.Separator(opts_frame, orient=tk.VERTICAL).pack(
        side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

    ttk.Label(opts_frame, text="Zoom:").pack(side=tk.LEFT)
    zoom_var = tk.StringVar(value="Fit Width")
    zoom_combo = ttk.Combobox(
        opts_frame, textvariable=zoom_var,
        values=['Fit Width', 'Fit Height', '50%', '75%', '100%',
                '125%', '150%', '200%'],
        width=10, state="normal",
    )
    zoom_combo.pack(side=tk.LEFT, padx=(4, 8))

    ttk.Separator(opts_frame, orient=tk.VERTICAL).pack(
        side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

    # External colored images directory
    external_dir_var = tk.StringVar(value="")
    ttk.Label(opts_frame, text="External Dir:").pack(side=tk.LEFT)
    external_dir_entry = ttk.Entry(opts_frame, textvariable=external_dir_var,
                                   width=20)
    external_dir_entry.pack(side=tk.LEFT, padx=(4, 2))
    attach_tooltip(external_dir_entry,
                   "Path to folder with pre-colored images for comparison.")

    def browse_external_dir():
        d = filedialog.askdirectory(title="Select External Colored Images Folder",
                                    parent=preview_window)
        if d:
            external_dir_var.set(d)

    ttk.Button(opts_frame, text="…", width=3,
               command=browse_external_dir).pack(side=tk.LEFT, padx=(0, 4))

    # --- Row 2: Notices (only shown when needed) ---
    preview_mode_notice_var = tk.StringVar(value="")
    notice_label = ttk.Label(main_frame, textvariable=preview_mode_notice_var,
                             style="Italic.TLabel")
    notice_label.grid(row=2, column=0, sticky="w", pady=(0, 2))

    image_pane = ttk.Panedwindow(main_frame, orient=tk.HORIZONTAL)
    image_pane.grid(row=4, column=0, sticky="nsew", pady=(0, 4))

    def create_image_panel(title):
        panel = ttk.Frame(image_pane, padding=(2, 2, 2, 2))
        panel.columnconfigure(0, weight=1)
        # Row 0: title, Row 1: canvas (gets all stretch), Row 2: x-scrollbar
        panel.rowconfigure(1, weight=1)

        title_frame = ttk.Frame(panel)
        title_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        
        ttk.Label(title_frame, text=title, font=("Helvetica", 11, "bold")).pack(side=tk.LEFT)

        canvas = tk.Canvas(panel, background="#101010", highlightthickness=1, highlightbackground="#c6c6c6")
        canvas.grid(row=1, column=0, sticky="nsew")

        y_scrollbar = ttk.Scrollbar(panel, orient="vertical")
        y_scrollbar.grid(row=1, column=1, sticky="ns")
        x_scrollbar = ttk.Scrollbar(panel, orient="horizontal")
        x_scrollbar.grid(row=2, column=0, sticky="ew")

        image_item = canvas.create_image(0, 0, anchor="nw")
        return panel, canvas, image_item, x_scrollbar, y_scrollbar, title_frame

    original_panel, original_canvas, original_image_id, original_x_scrollbar, original_y_scrollbar, orig_title_frame = create_image_panel("Original")
    colorized_panel, colorized_canvas, colorized_image_id, colorized_x_scrollbar, colorized_y_scrollbar, col_title_frame = create_image_panel("Colorized")
    image_pane.add(original_panel, weight=1)
    image_pane.add(colorized_panel, weight=1)

    # ---------------------------------------------------------
    # Professional Mask Editor Integration
    # ---------------------------------------------------------
    try:
        from Backend.preview_tools import MaskEditorWorkspace, tool_tooltip
        
        def on_user_mask_updated():
            self.config.user_mask = mask_editor.get_final_mask()
            schedule_preview_rerun("User mask manually updated", delay_ms=500)
        
        mask_editor = MaskEditorWorkspace(
            original_canvas, original_image_id,
            on_mask_changed_callback=on_user_mask_updated,
            pixel_wipe_active_check=pixel_wipe_var.get,
        )
        
        # --- Row 3: Mask tools + sliders in main_frame (NOT inside panels) ---
        mask_toolbar_frame = ttk.Frame(main_frame)
        mask_toolbar_frame.grid(row=3, column=0, sticky="ew", pady=(0, 2))

        # Left side: tool buttons
        tool_btn_frame = ttk.Frame(mask_toolbar_frame)
        tool_btn_frame.pack(side=tk.LEFT, padx=(0, 4))

        def _create_tool_btn(parent, tool_name, label, width=6):
            """Create a tool button with shortcut tooltip."""
            tip_text = tool_tooltip(tool_name)
            btn = ttk.Button(
                parent, text=label, width=width,
                command=lambda t=tool_name: mask_editor.set_tool(t)
            )
            btn.pack(side=tk.LEFT, padx=1)
            # Tooltip on hover
            def _enter(e, txt=tip_text):
                btn._tip = tk.Toplevel(btn)
                btn._tip.wm_overrideredirect(True)
                btn._tip.wm_geometry(f"+{e.x_root+10}+{e.y_root+10}")
                lbl = tk.Label(btn._tip, text=txt, background="#333",
                               foreground="#eee", font=("Segoe UI", 9),
                               padx=6, pady=2)
                lbl.pack()
            def _leave(e):
                if hasattr(btn, '_tip') and btn._tip:
                    btn._tip.destroy()
                    btn._tip = None
            btn.bind('<Enter>', _enter)
            btn.bind('<Leave>', _leave)
            return btn

        _create_tool_btn(tool_btn_frame, 'brush',           'Brush')
        _create_tool_btn(tool_btn_frame, 'eraser',          'Eraser')
        _create_tool_btn(tool_btn_frame, 'free_lasso',      'F.Lasso')
        _create_tool_btn(tool_btn_frame, 'poly_lasso',      'P.Lasso')
        _create_tool_btn(tool_btn_frame, 'rect_marquee',    'Rect')
        _create_tool_btn(tool_btn_frame, 'ellipse_marquee', 'Ellipse')
        _create_tool_btn(tool_btn_frame, 'magic_wand',      'Wand')

        ttk.Separator(tool_btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        def disable_tool():
            mask_editor.set_tool(None)
        def clear_mask():
            mask_editor.set_tool(None)
            if session.get('last_original_pil'):
                w, h = session['last_original_pil'].size
                zoom = session.get('wipe_zoom', 1.0)
                mask_editor.reset_mask(w, h, zoom)
                on_user_mask_updated()
        def do_undo():
            mask_editor.undo()
        def do_redo():
            mask_editor.redo()

        ttk.Button(tool_btn_frame, text="None", width=5, command=disable_tool).pack(side=tk.LEFT, padx=1)
        ttk.Button(tool_btn_frame, text="Clear", width=5, command=clear_mask).pack(side=tk.LEFT, padx=1)

        def reset_to_ocr():
            mask_editor.reset_to_base_mask()
        ttk.Button(tool_btn_frame, text="OCR Reset", width=8, command=reset_to_ocr).pack(side=tk.LEFT, padx=1)

        ttk.Button(tool_btn_frame, text="Undo", width=5, command=do_undo).pack(side=tk.LEFT, padx=1)
        ttk.Button(tool_btn_frame, text="Redo", width=5, command=do_redo).pack(side=tk.LEFT, padx=1)

        # Right side: sliders + overlay toggle
        ttk.Separator(mask_toolbar_frame, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=6, pady=2)

        brush_size_var = tk.IntVar(value=20)
        wand_tol_var = tk.IntVar(value=32)
        feather_var = tk.IntVar(value=0)

        def on_brush_size(val):
            mask_editor.brush_size = int(float(val))
        def on_wand_tol(val):
            mask_editor.wand_tolerance = int(float(val))
        def on_feather(val):
            mask_editor.feather_radius = int(float(val))

        ttk.Label(mask_toolbar_frame, text="Size:").pack(side=tk.LEFT, padx=(4, 0))
        brush_slider = ttk.Scale(mask_toolbar_frame, from_=1, to=200,
                                 variable=brush_size_var, orient=tk.HORIZONTAL,
                                 command=on_brush_size, length=80)
        brush_slider.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(mask_toolbar_frame, text="Tol:").pack(side=tk.LEFT)
        tol_slider = ttk.Scale(mask_toolbar_frame, from_=1, to=128,
                               variable=wand_tol_var, orient=tk.HORIZONTAL,
                               command=on_wand_tol, length=80)
        tol_slider.pack(side=tk.LEFT, padx=(0, 6))

        ttk.Label(mask_toolbar_frame, text="Feather:").pack(side=tk.LEFT)
        feather_slider = ttk.Scale(mask_toolbar_frame, from_=0, to=20,
                                   variable=feather_var, orient=tk.HORIZONTAL,
                                   command=on_feather, length=60)
        feather_slider.pack(side=tk.LEFT, padx=(0, 6))

        # Mask overlay toggle
        overlay_var = tk.BooleanVar(value=True)
        def toggle_overlay():
            mask_editor.show_mask_overlay = overlay_var.get()
            mask_editor._render_overlay()
        ttk.Checkbutton(mask_toolbar_frame, text="Mask",
                        variable=overlay_var, command=toggle_overlay).pack(side=tk.LEFT, padx=4)

        # Bind keyboard shortcuts to the preview window
        mask_editor.bind_keyboard_shortcuts(main_frame)
        
    except Exception as e:
        import traceback as _tb
        self.log(f"[!] Could not load manual mask tools: {e}")
        _tb.print_exc()
        mask_editor = None

    scroll_sync_guard = {'active': False}

    def update_x_scrollbars(first, last):
        original_x_scrollbar.set(first, last)
        colorized_x_scrollbar.set(first, last)

    def update_y_scrollbars(first, last):
        original_y_scrollbar.set(first, last)
        colorized_y_scrollbar.set(first, last)

    def xview_both(*args):
        if scroll_sync_guard['active']:
            return
        scroll_sync_guard['active'] = True
        try:
            original_canvas.xview(*args)
            colorized_canvas.xview(*args)
            first, last = original_canvas.xview()
            update_x_scrollbars(first, last)
        finally:
            scroll_sync_guard['active'] = False

    def yview_both(*args):
        if scroll_sync_guard['active']:
            return
        scroll_sync_guard['active'] = True
        try:
            original_canvas.yview(*args)
            colorized_canvas.yview(*args)
            first, last = original_canvas.yview()
            update_y_scrollbars(first, last)
        finally:
            scroll_sync_guard['active'] = False

    original_canvas.configure(xscrollcommand=update_x_scrollbars, yscrollcommand=update_y_scrollbars)
    colorized_canvas.configure(xscrollcommand=update_x_scrollbars, yscrollcommand=update_y_scrollbars)
    original_x_scrollbar.configure(command=xview_both)
    colorized_x_scrollbar.configure(command=xview_both)
    original_y_scrollbar.configure(command=yview_both)
    colorized_y_scrollbar.configure(command=yview_both)

    def bind_canvas_scroll(canvas):
        def _on_mousewheel(event):
            ctrl_pressed = (event.state & 0x0004) != 0
            if ctrl_pressed:
                zoom_direction = 1 if (event.num == 4 or event.delta > 0) else -1
                return change_zoom_by_step(zoom_direction)

            if event.num == 4:
                units = -1
            elif event.num == 5:
                units = 1
            elif event.delta:
                units = int(-event.delta / 120)
                if units == 0:
                    units = -1 if event.delta > 0 else 1
            else:
                return "break"

            if event.state & 0x0001:
                xview_both('scroll', units, 'units')
            else:
                yview_both('scroll', units, 'units')
            return "break"

        canvas.bind("<MouseWheel>", _on_mousewheel, add="+")
        canvas.bind("<Shift-MouseWheel>", _on_mousewheel, add="+")
        canvas.bind("<Button-4>", _on_mousewheel, add="+")
        canvas.bind("<Button-5>", _on_mousewheel, add="+")

    bind_canvas_scroll(original_canvas)
    bind_canvas_scroll(colorized_canvas)

    preset_frame = ttk.LabelFrame(main_frame, text="Preset", padding="8")
    preset_frame.grid(row=5, column=0, sticky="ew", pady=(0, 6))
    preset_frame.columnconfigure(1, weight=1)

    ttk.Label(preset_frame, text="Selected:").grid(row=0, column=0, sticky="w", padx=(0, 6))
    preset_name_var = tk.StringVar(value=active_folder_preset)
    preset_selector = ttk.Combobox(preset_frame, textvariable=preset_name_var, state="readonly")
    preset_selector.grid(row=0, column=1, sticky="ew")
    preset_actions_button = ttk.Menubutton(preset_frame, text="Actions ▼")
    preset_actions_button.grid(row=0, column=2, padx=(6, 0), sticky="e")

    tuning_toggle_button = ttk.Button(main_frame, text="Show Live Tuning")
    tuning_toggle_button.grid(row=6, column=0, sticky="w", pady=(0, 4))

    tuning_frame = ttk.LabelFrame(main_frame, text="Live Tuning", padding="8")
    tuning_frame.columnconfigure(0, weight=1)
    tuning_frame.columnconfigure(1, weight=1)
    tuning_frame_shown = {'value': False}

    left_tuning = ttk.Frame(tuning_frame)
    left_tuning.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
    left_tuning.columnconfigure(0, weight=1)
    right_tuning = ttk.Frame(tuning_frame)
    right_tuning.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
    right_tuning.columnconfigure(0, weight=1)

    control_refreshers = []

    def create_compact_slider(parent, label_text, variable, from_value, to_value, tooltip_text):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)

        label = ttk.Label(row, text=label_text, width=21)
        label.pack(side=tk.LEFT)

        slider = ttk.Scale(row, from_=from_value, to=to_value, orient="horizontal")
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 6))

        value_var = tk.StringVar()
        value_label = ttk.Label(row, textvariable=value_var, width=4, anchor="e")
        value_label.pack(side=tk.LEFT)

        def refresh_value(*_):
            value_var.set(str(int(float(variable.get()))))

        def on_slider_change(value):
            variable.set(int(float(value)))
            refresh_value()

        slider.configure(variable=variable, command=on_slider_change)
        refresh_value()
        control_refreshers.append(refresh_value)

        attach_tooltip(label, tooltip_text)
        attach_tooltip(slider, tooltip_text)
        return slider

    denoise_sigma_slider = create_compact_slider(
        left_tuning,
        "Denoise Sigma",
        self.denoise_sigma,
        1,
        100,
        "Denoiser strength before colorization. Higher values remove more grain but can soften details.",
    )

    chroma_row = ttk.Frame(left_tuning)
    chroma_row.pack(fill=tk.X, pady=2)
    chroma_label = ttk.Label(chroma_row, text="Chroma Mode", width=21)
    chroma_label.pack(side=tk.LEFT)
    chroma_combo = ttk.Combobox(
        chroma_row,
        textvariable=self.chroma_resize_mode,
        values=['BICUBIC', 'BILINEAR', 'LANCZOS'],
        state="readonly",
        width=12,
    )
    chroma_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
    attach_tooltip(chroma_label, "Resampling method for chroma resize in luminance transfer.")
    attach_tooltip(chroma_combo, "Resampling method for chroma resize in luminance transfer.")

    edge_check = ttk.Checkbutton(left_tuning, text="Edge Protect", variable=self.edge_chroma_protection)
    edge_check.pack(anchor="w", pady=(4, 0))
    attach_tooltip(edge_check, "Reduces color bleed near strong line edges.")
    edge_strength_slider = create_compact_slider(
        left_tuning,
        "Edge Strength",
        self.edge_chroma_strength,
        0,
        100,
        "How strongly edge-aware chroma suppression is applied.",
    )

    ink_check = ttk.Checkbutton(right_tuning, text="Line/Ink Protect", variable=self.line_ink_protection)
    ink_check.pack(anchor="w", pady=(0, 0))
    attach_tooltip(ink_check, "Suppresses chroma in dark line-art regions to reduce color bleed around strokes.")
    ink_strength_slider = create_compact_slider(
        right_tuning,
        "Line/Ink Strength",
        self.line_ink_protection_strength,
        0,
        100,
        "How strongly line-art chroma suppression is applied.",
    )

    screentone_check = ttk.Checkbutton(right_tuning, text="Screentone Smooth", variable=self.screentone_chroma_smoothing)
    screentone_check.pack(anchor="w", pady=(4, 0))
    attach_tooltip(screentone_check, "Smooths chroma in screentone/high-frequency regions to reduce speckle.")
    screentone_strength_slider = create_compact_slider(
        right_tuning,
        "Tone Strength",
        self.screentone_smoothing_strength,
        0,
        100,
        "How strongly screentone chroma smoothing is applied.",
    )

    debug_mask_type_var = tk.StringVar(value="None")
    debug_mask_mode_var = tk.StringVar(value="Overlay")
    debug_mask_opacity_var = tk.IntVar(value=55)

    mask_type_row = ttk.Frame(left_tuning)
    mask_type_row.pack(fill=tk.X, pady=(6, 2))
    mask_type_label = ttk.Label(mask_type_row, text="Debug Mask", width=21)
    mask_type_label.pack(side=tk.LEFT)
    mask_type_combo = ttk.Combobox(
        mask_type_row,
        textvariable=debug_mask_type_var,
        values=list(DEBUG_MASK_UI_VALUES),
        state="readonly",
        width=12,
    )
    mask_type_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
    attach_tooltip(mask_type_label, "Highlight where detection heuristics activate.")
    attach_tooltip(mask_type_combo, "Select which detection mask to visualize.")

    mask_mode_row = ttk.Frame(left_tuning)
    mask_mode_row.pack(fill=tk.X, pady=2)
    mask_mode_label = ttk.Label(mask_mode_row, text="Mask View", width=21)
    mask_mode_label.pack(side=tk.LEFT)
    mask_mode_combo = ttk.Combobox(
        mask_mode_row,
        textvariable=debug_mask_mode_var,
        values=['Overlay', 'Mask only'],
        state="readonly",
        width=12,
    )
    mask_mode_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
    attach_tooltip(mask_mode_label, "Overlay paints mask on output; Mask only renders mask over blank background.")
    attach_tooltip(mask_mode_combo, "Choose visualization style for debug mask.")

    mask_opacity_slider = create_compact_slider(
        left_tuning,
        "Mask Opacity",
        debug_mask_opacity_var,
        5,
        100,
        "Overlay opacity for mask visualization.",
    )

    session = {
        'window': preview_window,
        'image_folder': image_folder,
        'image_paths': image_paths,
        'current_index': 0,
        'current_path': image_paths[0],
        'history': deque(maxlen=40),
        'generation': 0,
        'compare_mode_active': False,
        'worker_running': False,
        'scheduled_after': None,
        'source_cache': OrderedDict(),
        'denoise_cache': OrderedDict(),
        'trace_handles': [],
        'pending_reset_scroll': True,
        'last_original_pil': None,
        'last_colorized_pil': None,
        'last_source_np': None,
        'last_luminance_source_np': None,
        'last_colorized_np': None,
        'debug_masks_cache': None,
        'last_render_status_base': '',
        'status_var': status_var,
        'mode_notice_var': preview_mode_notice_var,
        'schedule_preview_rerun': None,
        'load_compare_preview': None,
        'update_runtime_mode': None,
        'original_canvas': original_canvas,
        'colorized_canvas': colorized_canvas,
        'original_image_id': original_image_id,
        'colorized_image_id': colorized_image_id,
        'mask_editor': mask_editor,
        'mask_cache': {},  # Per-image mask cache: {image_path: mask_array}
    }
    self.preview_session = session

    def window_is_alive():
        try:
            return bool(preview_window.winfo_exists())
        except tk.TclError:
            return False

    def is_batch_running_for_preview():
        return self._is_batch_processing_active()

    def resolve_output_image_path(source_image_path):
        output_root = self.output_folder.get().strip().strip("'\"")
        if not output_root:
            return ''

        try:
            rel_path = os.path.relpath(source_image_path, session['image_folder'])
        except Exception:
            rel_path = os.path.basename(source_image_path)

        return os.path.normpath(os.path.join(output_root, rel_path))

    def load_compare_preview_for_current(reset_scroll=False):
        image_path = session.get('current_path')
        if not image_path or not os.path.exists(image_path):
            return False

        try:
            input_limit = sanitize_input_width_limit(getattr(self.config, 'input_image_size', DEFAULT_INPUT_WIDTH))
            with PIL.Image.open(image_path) as source_handle:
                source_np = np.array(source_handle.convert("RGB"))
            source_np, _, _, _ = clamp_image_to_input_width(source_np, input_limit)

            output_path = None
            output_exists = False
            external_loaded = False

            # Try external colored dir first
            ext_dir = external_dir_var.get().strip()
            both_active = bool(self.config.denoise and
                               getattr(self.config, 'colorize', True))
            if ext_dir and os.path.isdir(ext_dir) and not both_active:
                base = os.path.basename(image_path)
                stem = os.path.splitext(base)[0]
                # Try exact match, then common extensions
                for ext in ['', '.png', '.jpg', '.jpeg', '.webp']:
                    candidate = os.path.join(ext_dir, stem + ext) if ext else os.path.join(ext_dir, base)
                    if os.path.isfile(candidate):
                        output_path = candidate
                        output_exists = True
                        external_loaded = True
                        break

            # Fall back to pipeline output
            if not output_exists:
                output_path = resolve_output_image_path(image_path)
                output_exists = bool(output_path and os.path.exists(output_path))

            if output_exists:
                with PIL.Image.open(output_path) as output_handle:
                    output_np = np.array(output_handle.convert("RGB"))
            else:
                output_np = source_np.copy()

            session['last_original_pil'] = PIL.Image.fromarray(source_np)
            session['last_source_np'] = source_np.copy()
            session['last_luminance_source_np'] = source_np.copy()
            session['last_colorized_np'] = output_np.copy()
            session['last_colorized_pil'] = PIL.Image.fromarray(output_np)
            if external_loaded:
                status_text = f"External: {os.path.basename(output_path)}"
            elif output_exists:
                status_text = "Compare: input vs pipeline output"
            else:
                status_text = "Compare: output not generated yet"
            session['last_render_status_base'] = status_text
            invalidate_debug_mask_cache()
            refresh_rendered_images(reset_scroll=reset_scroll)

            if mask_editor:
                zoom = session.get('wipe_zoom', 1.0)
                # Set grayscale source for magic wand tool
                import cv2 as _cv2
                gray = _cv2.cvtColor(source_np, _cv2.COLOR_RGB2GRAY)
                mask_editor.set_source_image(gray)
                # Check per-image mask cache first
                cached_mask = session.get('mask_cache', {}).get(image_path)
                if cached_mask is not None and cached_mask.shape[:2] == source_np.shape[:2]:
                    mask_editor.load_mask(cached_mask, zoom)
                else:
                    user_mask = getattr(self.config, 'user_mask', None)
                    if user_mask is not None and user_mask.shape[:2] == source_np.shape[:2]:
                        mask_editor.load_mask(user_mask, zoom)
                    else:
                        mask_editor.reset_mask(source_np.shape[1], source_np.shape[0], zoom)

            status_var.set(status_text)
            return True
        except Exception as err:
            self.log(f"[!] Compare preview failed for {os.path.basename(image_path)}: {err}")
            status_var.set("Compare preview failed. Check logs for details.")
            return False

    def lru_get(cache, key):
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
        return None

    def lru_put(cache, key, value, max_size):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > max_size:
            cache.popitem(last=False)

    zoom_steps = [25, 33, 50, 67, 75, 90, 100, 110, 125, 150, 175, 200, 250, 300, 400]

    def normalize_zoom_value(raw_value):
        text = str(raw_value or '').strip()
        lowered = text.lower().replace('_', ' ').replace('-', ' ')
        compact = ''.join(lowered.split())

        if lowered in {'fit width', 'fit to width'} or compact in {'fitwidth', 'fwidth', 'width'}:
            return 'fit_width', 'Fit Width'
        if lowered in {'fit height', 'fit to height'} or compact in {'fitheight', 'fheight', 'height'}:
            return 'fit_height', 'Fit Height'

        if text.endswith('%'):
            text = text[:-1].strip()

        try:
            percent = int(round(float(text)))
        except ValueError:
            percent = 100

        percent = max(10, min(400, percent))
        return 'percent', f"{percent}%"

    def current_zoom_mode():
        mode, normalized = normalize_zoom_value(zoom_var.get())
        if zoom_var.get().strip() != normalized:
            zoom_var.set(normalized)
        return mode, normalized

    def zoom_factor_for_canvas(canvas, pil_image):
        mode, normalized = current_zoom_mode()
        if mode == 'fit_width':
            # Always use original_canvas width for both panels
            # to ensure identical zoom in side-by-side view
            viewport = max(1, original_canvas.winfo_width() - 6)
            return max(0.05, min(8.0, viewport / max(1, pil_image.width)))
        if mode == 'fit_height':
            # Use the SMALLER height (colorized canvas has more
            # vertical space since original has toolbar rows)
            h1 = max(1, original_canvas.winfo_height() - 6)
            h2 = max(1, colorized_canvas.winfo_height() - 6)
            viewport = min(h1, h2)
            return max(0.05, min(8.0, viewport / max(1, pil_image.height)))

        percent = int(normalized[:-1])
        return max(0.05, min(8.0, percent / 100.0))

    def effective_zoom_percent():
        mode, normalized = normalize_zoom_value(zoom_var.get())
        if mode == 'fit_width' and session['last_original_pil'] is not None:
            return int(round(100.0 * max(1, original_canvas.winfo_width() - 6) / max(1, session['last_original_pil'].width)))
        if mode == 'fit_height' and session['last_original_pil'] is not None:
            return int(round(100.0 * max(1, original_canvas.winfo_height() - 6) / max(1, session['last_original_pil'].height)))
        return int(normalized[:-1]) if normalized.endswith('%') else 100

    def change_zoom_by_step(direction):
        current_percent = max(10, min(400, effective_zoom_percent()))

        if direction > 0:
            candidates = [value for value in zoom_steps if value > current_percent]
            next_percent = candidates[0] if candidates else min(400, current_percent + 10)
        else:
            candidates = [value for value in zoom_steps if value < current_percent]
            next_percent = candidates[-1] if candidates else max(10, current_percent - 10)

        zoom_var.set(f"{next_percent}%")
        refresh_rendered_images(reset_scroll=False)
        return "break"

    def render_canvas_image(canvas, image_id, pil_image, wipe_pil=None, is_original=True):
        zoom_factor = zoom_factor_for_canvas(canvas, pil_image)
        
        cache_key_prefix = 'orig' if is_original else 'color'
        cache_zoom_key = f"{cache_key_prefix}_zoom"
        cache_img_key = f"{cache_key_prefix}_img"
        
        if session.get(cache_zoom_key) == zoom_factor and session.get(cache_img_key) is not None:
            display_image = session[cache_img_key]
        else:
            if abs(zoom_factor - 1.0) > 1e-6:
                resized_w = max(1, int(round(pil_image.width * zoom_factor)))
                resized_h = max(1, int(round(pil_image.height * zoom_factor)))
                resample_mode = PIL.Image.Resampling.LANCZOS if zoom_factor < 1.0 else PIL.Image.Resampling.BICUBIC
                display_image = pil_image.resize((resized_w, resized_h), resample_mode)
            else:
                display_image = pil_image.copy()
            session[cache_zoom_key] = zoom_factor
            session[cache_img_key] = display_image

        # Always sync zoom to mask editor for original panel
        if is_original and mask_editor:
            mask_editor.update_overlay_zoom(zoom_factor)

        if wipe_pil is not None:
            if session.get('wipe_zoom') == zoom_factor and session.get('wipe_img') is not None:
                wipe_display = session['wipe_img']
            else:
                if abs(zoom_factor - 1.0) > 1e-6:
                    resized_w = max(1, int(round(wipe_pil.width * zoom_factor)))
                    resized_h = max(1, int(round(wipe_pil.height * zoom_factor)))
                    resample_mode = PIL.Image.Resampling.LANCZOS if zoom_factor < 1.0 else PIL.Image.Resampling.BICUBIC
                    wipe_display = wipe_pil.resize((resized_w, resized_h), resample_mode)
                else:
                    wipe_display = wipe_pil.copy()
                session['wipe_zoom'] = zoom_factor
                session['wipe_img'] = wipe_display
                if mask_editor:
                    mask_editor.update_overlay_zoom(zoom_factor)

            wipe_x = int(display_image.width * session.get('wipe_percent', 0.5))
            composite = display_image.copy()
            if wipe_x > 0:
                left_crop = wipe_display.crop((0, 0, wipe_x, wipe_display.height))
                composite.paste(left_crop, (0, 0))
            draw = PIL.ImageDraw.Draw(composite)
            draw.line([(wipe_x, 0), (wipe_x, composite.height)], fill=(255, 255, 255), width=2)
            display_image = composite

        photo = ImageTk.PhotoImage(display_image)
        canvas.itemconfigure(image_id, image=photo)
        canvas.image = photo
        canvas.config(scrollregion=(0, 0, display_image.width, display_image.height))

    def refresh_rendered_images(reset_scroll=False):
        try:
            # Always sync the PanedWindow layout state
            if pixel_wipe_var.get():
                if colorized_panel.winfo_manager():
                    image_pane.forget(colorized_panel)
            else:
                if not colorized_panel.winfo_manager():
                    image_pane.add(colorized_panel, weight=1)

            if session['last_original_pil'] is None or session['last_colorized_pil'] is None:
                return

            if reset_scroll:
                x_anchor = 0.0
                y_anchor = 0.0
            else:
                x_anchor = original_canvas.xview()[0] if original_canvas.xview() else 0.0
                y_anchor = original_canvas.yview()[0] if original_canvas.yview() else 0.0

            if pixel_wipe_var.get():
                render_canvas_image(
                    session['original_canvas'],
                    session['original_image_id'],
                    session['last_original_pil'],
                    wipe_pil=session['last_colorized_pil'],
                    is_original=True
                )
            else:
                if not colorized_panel.winfo_manager():
                    image_pane.add(colorized_panel, weight=1)
                render_canvas_image(
                    session['original_canvas'],
                    session['original_image_id'],
                    session['last_original_pil'],
                    is_original=True
                )
                render_canvas_image(
                    session['colorized_canvas'],
                    session['colorized_image_id'],
                    session['last_colorized_pil'],
                    is_original=False
                )

            xview_both('moveto', x_anchor)
            yview_both('moveto', y_anchor)
        except Exception as e:
            import traceback
            traceback.print_exc()
            status_var.set(f"UI Error: {str(e)}")

    def on_wipe_motion(event):
        if pixel_wipe_var.get() and getattr(event.widget, '_wipe_dragging', False):
            canvas_x = event.widget.canvasx(event.x)
            region = event.widget.cget("scrollregion")
            if region:
                parts = region.split()
                if len(parts) == 4:
                    w = float(parts[2])
                    percent = max(0.0, min(1.0, canvas_x / w))
                    session['wipe_percent'] = percent
                    refresh_rendered_images()

    def on_wipe_press(event):
        if pixel_wipe_var.get():
            # Don't start wipe drag if a mask tool is active
            if mask_editor and mask_editor.active_tool:
                return
            event.widget._wipe_dragging = True
            on_wipe_motion(event)

    def on_wipe_release(event):
        event.widget._wipe_dragging = False

    original_canvas.bind("<ButtonPress-1>", on_wipe_press, add="+")
    original_canvas.bind("<B1-Motion>", on_wipe_motion, add="+")
    original_canvas.bind("<ButtonRelease-1>", on_wipe_release, add="+")
    
    pixel_wipe_var.trace_add('write', lambda *_: refresh_rendered_images())

    def current_debug_mask_snapshot():
        return {
            'type': debug_mask_label_to_key(debug_mask_type_var.get()),
            'mode': 'mask_only' if debug_mask_mode_var.get().strip().lower().startswith('mask') else 'overlay',
            'opacity': max(0.05, min(1.0, debug_mask_opacity_var.get() / 100.0)),
        }

    def invalidate_debug_mask_cache():
        session['debug_masks_cache'] = None

    def get_cached_debug_masks():
        luma_np = session.get('last_luminance_source_np')
        if luma_np is None:
            return None

        cache_blob = session.get('debug_masks_cache')
        if (
            isinstance(cache_blob, dict)
            and cache_blob.get('shape') == luma_np.shape[:2]
        ):
            return cache_blob.get('masks')

        debug_masks = compute_transfer_debug_masks(luma_np)

        session['debug_masks_cache'] = {
            'shape': luma_np.shape[:2],
            'masks': debug_masks,
        }
        return debug_masks

    def apply_debug_visual_from_cache(reset_scroll=False):
        base_colorized_np = session.get('last_colorized_np')
        if base_colorized_np is None:
            return False

        colorized_np = base_colorized_np.copy()
        debug_mask_note = ''
        debug_snapshot = current_debug_mask_snapshot()
        selected_mask = debug_snapshot['type']

        if selected_mask in DEBUG_MASK_COLOR_MAP:
            debug_masks = get_cached_debug_masks()
            selected_debug_mask = debug_masks.get(selected_mask) if debug_masks else None
            if selected_debug_mask is not None:
                selected_debug_mask = np.clip(selected_debug_mask, 0.0, 1.0)
                colorized_np = apply_debug_mask_visual(
                    colorized_np,
                    selected_debug_mask,
                    mode=debug_snapshot['mode'],
                    color=DEBUG_MASK_COLOR_MAP[selected_mask],
                    opacity=debug_snapshot['opacity'],
                )
                debug_mask_note = f" mask:{selected_mask}/{debug_snapshot['mode']}-full"

        session['last_colorized_pil'] = PIL.Image.fromarray(colorized_np)
        session['color_img'] = None
        session['wipe_img'] = None
        refresh_rendered_images(reset_scroll=reset_scroll)

        base_status = session.get('last_render_status_base', '').strip()
        status_var.set(f"{base_status}{debug_mask_note}" if base_status else f"Preview{debug_mask_note}")
        return True

    def refresh_image_caption():
        index = session['current_index']
        image_path = session['image_paths'][index]
        relative_name = os.path.relpath(image_path, session['image_folder'])
        image_info_var.set(f"Image {index + 1}/{len(session['image_paths'])}: {relative_name}")

    def refresh_preset_selector(selected_name=None):
        values = self._ordered_preset_names()
        preset_selector['values'] = values

        fallback = self._active_quality_preset_for_folder(image_folder)
        if selected_name in self.quality_presets:
            preset_name_var.set(selected_name)
        elif fallback in self.quality_presets:
            preset_name_var.set(fallback)
        else:
            preset_name_var.set('Default')

    session['refresh_preset_selector'] = refresh_preset_selector

    tuning_widgets = [
        denoise_sigma_slider,
        chroma_combo,
        edge_check,
        edge_strength_slider,
        ink_check,
        ink_strength_slider,
        screentone_check,
        screentone_strength_slider,
        mask_type_combo,
        mask_mode_combo,
        mask_opacity_slider,
    ]
    session['tuning_widgets'] = tuning_widgets

    def set_tuning_controls_enabled(enabled):
        desired_state = ["!disabled"] if enabled else ["disabled"]
        for widget in tuning_widgets:
            try:
                widget.state(desired_state)
            except (AttributeError, tk.TclError):
                continue

    def sync_tuning_states(*_):
        if is_batch_running_for_preview():
            set_tuning_controls_enabled(False)
            return

        if self.edge_chroma_protection.get():
            edge_strength_slider.state(["!disabled"])
        else:
            edge_strength_slider.state(["disabled"])

        if self.line_ink_protection.get():
            ink_strength_slider.state(["!disabled"])
        else:
            ink_strength_slider.state(["disabled"])

        if self.screentone_chroma_smoothing.get():
            screentone_strength_slider.state(["!disabled"])
        else:
            screentone_strength_slider.state(["disabled"])

    def update_preview_runtime_mode():
        running = is_batch_running_for_preview()
        if running:
            if not bool(session.get('compare_mode_active', False)):
                session['compare_mode_active'] = True
                session['generation'] += 1
                if session['scheduled_after'] is not None:
                    try:
                        preview_window.after_cancel(session['scheduled_after'])
                    except tk.TclError:
                        pass
                    session['scheduled_after'] = None

            preview_mode_notice_var.set(self._preview_running_warning_text())
            tuning_toggle_button.state(["disabled"])
            tuning_toggle_button.config(text="Live Tuning Disabled While Running")
            set_tuning_controls_enabled(False)
            load_compare_preview_for_current(reset_scroll=session.get('pending_reset_scroll', False))
            session['pending_reset_scroll'] = False
        else:
            was_compare_mode = bool(session.get('compare_mode_active', False))
            session['compare_mode_active'] = False
            preview_mode_notice_var.set("")
            tuning_toggle_button.state(["!disabled"])
            tuning_toggle_button.config(text="Hide Live Tuning" if tuning_frame_shown['value'] else "Show Live Tuning")
            set_tuning_controls_enabled(True)
            sync_tuning_states()
            if was_compare_mode:
                session['pending_reset_scroll'] = False
                schedule_preview_rerun(reason="Live preview resumed", delay_ms=0)

    session['update_runtime_mode'] = update_preview_runtime_mode
    session['load_compare_preview'] = load_compare_preview_for_current

    def toggle_tuning():
        if is_batch_running_for_preview():
            update_preview_runtime_mode()
            return

        if tuning_frame_shown['value']:
            tuning_frame.grid_remove()
            tuning_toggle_button.config(text="Show Live Tuning")
            tuning_frame_shown['value'] = False
        else:
            tuning_frame.grid(row=7, column=0, sticky="ew", pady=(0, 0))
            tuning_toggle_button.config(text="Hide Live Tuning")
            tuning_frame_shown['value'] = True

    def set_current_index(new_index, reason, immediate=False):
        total = len(session['image_paths'])
        if total == 0:
            return

        # Save current mask to per-image cache before navigating away
        _me = session.get('mask_editor')
        if _me:
            _current_path = session.get('current_path')
            _current_mask = _me.get_mask_array()
            if _current_path and _current_mask is not None:
                import numpy as _np_cache
                if _np_cache.any(_current_mask):
                    session['mask_cache'][_current_path] = _current_mask.copy()
                elif _current_path in session['mask_cache']:
                    del session['mask_cache'][_current_path]

        normalized_index = new_index % total
        session['current_index'] = normalized_index
        session['current_path'] = session['image_paths'][normalized_index]
        if not session['history'] or session['history'][-1] != normalized_index:
            session['history'].append(normalized_index)
        session['pending_reset_scroll'] = True

        refresh_image_caption()
        rerun_delay = 0 if immediate else PREVIEW_RERUN_DEBOUNCE_MS
        schedule_preview_rerun(reason=reason, delay_ms=rerun_delay)

    def start_preview_worker_if_idle(trigger_reason):
        if not window_is_alive() or session['worker_running']:
            return

        if is_batch_running_for_preview():
            update_preview_runtime_mode()
            return

        session['worker_running'] = True
        run_generation = session['generation']
        image_path = session['current_path']
        status_var.set(f"{trigger_reason}...")

        def worker():
            source_cache_hit = False
            denoise_cache_hit = False
            try:
                if is_batch_running_for_preview():
                    self.after(0, update_preview_runtime_mode)
                    return

                self._update_config_from_gui()
                precision_override = None
                if bool(self.preview_fast_mode.get()) and supports_bf16():
                    precision_override = 'bf16'
                colorizer, _, denoiser = self._get_and_manage_models(
                    precision_policy_override=precision_override,
                )

                def run_with_torch_threads(thread_count, fn):
                    prev_threads = None
                    prev_interop = None
                    try:
                        prev_threads = torch.get_num_threads()
                    except Exception:
                        prev_threads = None
                    try:
                        prev_interop = torch.get_num_interop_threads()
                    except Exception:
                        prev_interop = None

                    try:
                        if thread_count:
                            torch.set_num_threads(max(1, int(thread_count)))
                            if prev_interop is not None:
                                torch.set_num_interop_threads(max(1, min(prev_interop, int(thread_count))))
                    except Exception:
                        pass

                    try:
                        return fn()
                    finally:
                        if prev_threads is not None:
                            try:
                                torch.set_num_threads(prev_threads)
                            except Exception:
                                pass
                        if prev_interop is not None:
                            try:
                                torch.set_num_interop_threads(prev_interop)
                            except Exception:
                                pass

                if not colorizer:
                    raise RuntimeError('Colorizer model could not be loaded.')

                started_at = time.time()
                input_limit = sanitize_input_width_limit(getattr(self.config, 'input_image_size', DEFAULT_INPUT_WIDTH))
                decode_key = (image_path, input_limit)

                source_np = lru_get(session['source_cache'], decode_key)
                if source_np is None:
                    with PIL.Image.open(image_path) as source_handle:
                        decoded = np.array(source_handle.convert("RGB"))

                    decoded, _, _, _ = clamp_image_to_input_width(decoded, input_limit)

                    lru_put(session['source_cache'], decode_key, decoded.copy(), PREVIEW_SOURCE_CACHE_SIZE)
                    source_np = decoded
                else:
                    source_cache_hit = True
                    source_np = source_np.copy()

                working_np = source_np.copy()
                if self.config.denoise and denoiser:
                    denoise_key = (decode_key, int(self.config.denoise_sigma))
                    cached_denoised = lru_get(session['denoise_cache'], denoise_key)
                    if cached_denoised is not None:
                        denoise_cache_hit = True
                        working_np = cached_denoised.copy()
                    else:
                        denoise_threads = THREADS_PER_INSTANCE.get('denoise', 1)
                        working_np = run_with_torch_threads(
                            denoise_threads,
                            lambda: denoiser.denoise(
                                working_np,
                                self.config.denoise_sigma,
                                image_name=os.path.basename(image_path),
                            ),
                        )
                        lru_put(session['denoise_cache'], denoise_key, working_np.copy(), PREVIEW_DENOISE_CACHE_SIZE)

                luminance_source = working_np.copy()

                # Pre-detect OCR on grayscale source BEFORE colorization.
                # OCR only needs L_source (grayscale), available now. By running
                # it first, the result gets cached and transfer_luminance_from_source
                # will get a cache hit — no GPU contention with colorizer.
                def ocr_status_cb(msg):
                    if window_is_alive() and run_generation == session['generation']:
                        self.after(0, lambda: status_var.set(msg))

                self.config.ocr_status_callback = ocr_status_cb
                self.config.current_image_path = image_path
                try:
                    from Backend.ocr_engine import get_detector
                    import cv2 as _cv2_pre
                    _pre_det = get_detector()
                    _pre_det.detect(
                        _cv2_pre.cvtColor(luminance_source, _cv2_pre.COLOR_RGB2GRAY)
                        if luminance_source.ndim == 3 else luminance_source,
                        self.config,
                    )
                    # Free OCR VRAM only if colorizer needs room
                    _pre_det.ensure_vram(needed_mb=2000)
                except Exception:
                    pass

                target_width = self.config.colorized_image_size
                if getattr(self.config, 'force_safe_colorizer_width', False):
                    target_width = 576

                original_width = working_np.shape[1]
                effective_width = min(original_width, target_width)
                adjusted_width = effective_width - (effective_width % 32)
                if adjusted_width == 0:
                    adjusted_width = 32

                def run_colorize():
                    colorizer.set_image(
                        working_np,
                        adjusted_width,
                        image_name=os.path.basename(image_path),
                    )
                    return colorizer.colorize()

                colorize_threads = THREADS_PER_INSTANCE.get('colorize', 1)
                colorized_np = run_with_torch_threads(colorize_threads, run_colorize)

                # OCR was pre-detected above; transfer_luminance will use cached result
                colorized_np = transfer_luminance_from_source(luminance_source, colorized_np, self.config)
                base_colorized_np = colorized_np.copy()

                original_display = PIL.Image.fromarray(source_np)
                render_duration = time.time() - started_at

                def update_gui_if_latest():
                    if not window_is_alive() or run_generation != session['generation']:
                        return

                    session['last_original_pil'] = original_display
                    session['last_source_np'] = source_np.copy()
                    session['last_luminance_source_np'] = luminance_source.copy()
                    session['last_colorized_np'] = base_colorized_np.copy()
                    session['orig_img'] = None
                    session['color_img'] = None
                    session['wipe_img'] = None
                    invalidate_debug_mask_cache()

                    cache_note = []
                    if source_cache_hit:
                        cache_note.append('decode-cache')
                    if denoise_cache_hit:
                        cache_note.append('denoise-cache')
                    cache_suffix = f" [{', '.join(cache_note)}]" if cache_note else ""
                    session['last_render_status_base'] = f"Rendered in {render_duration:.2f}s{cache_suffix}"
                    apply_debug_visual_from_cache(reset_scroll=session.get('pending_reset_scroll', False))
                    session['pending_reset_scroll'] = False

                    # Feed OCR combined_mask to mask editor as base_mask
                    _me = session.get('mask_editor')
                    if _me:
                        try:
                            import cv2 as _cv2_hook
                            from Backend.ocr_engine import get_detector
                            _det = get_detector()
                            _cached = _det._cache.get(
                                _det._fast_hash(
                                    _cv2_hook.cvtColor(source_np, _cv2_hook.COLOR_RGB2GRAY)
                                )
                            )
                            if _cached and _cached.combined_mask is not None:
                                zoom = session.get('wipe_zoom', 1.0)
                                _me.zoom_factor = zoom
                                _me.set_base_mask(_cached.combined_mask)
                        except Exception:
                            pass

                self.after(0, update_gui_if_latest)

            except Exception as err:
                self.log(f"[!!!] Failed to create preview for {os.path.basename(image_path)}. Error: {err}")

                def update_error_state():
                    if window_is_alive() and run_generation == session['generation']:
                        status_var.set("Preview failed. Check logs for details.")

                self.after(0, update_error_state)

            finally:
                def finalize_worker():
                    if not window_is_alive():
                        return
                    session['worker_running'] = False
                    if session['generation'] > run_generation:
                        start_preview_worker_if_idle("Applying latest changes")

                self.after(0, finalize_worker)

        threading.Thread(target=worker, daemon=True).start()

    def schedule_preview_rerun(reason, delay_ms=PREVIEW_RERUN_DEBOUNCE_MS):
        if not window_is_alive():
            return

        if session['scheduled_after'] is not None:
            try:
                preview_window.after_cancel(session['scheduled_after'])
            except tk.TclError:
                pass
            session['scheduled_after'] = None

        if is_batch_running_for_preview():
            update_preview_runtime_mode()
            return

        session['generation'] += 1

        def trigger_rerun():
            session['scheduled_after'] = None
            start_preview_worker_if_idle(reason)

        session['scheduled_after'] = preview_window.after(max(0, delay_ms), trigger_rerun)

    session['schedule_preview_rerun'] = schedule_preview_rerun

    def on_fast_mode_change(*_):
        if is_batch_running_for_preview():
            update_preview_runtime_mode()
            return
        session['pending_reset_scroll'] = False
        schedule_preview_rerun(reason="Fast mode", delay_ms=0)

    fast_mode_trace = self.preview_fast_mode.trace_add('write', on_fast_mode_change)
    session['trace_handles'].append((self.preview_fast_mode, fast_mode_trace))

    def on_live_control_change(*_):
        if is_batch_running_for_preview():
            update_preview_runtime_mode()
            return

        for refresher in control_refreshers:
            refresher()
        sync_tuning_states()
        session['pending_reset_scroll'] = False
        schedule_preview_rerun(reason="Live tuning", delay_ms=PREVIEW_RERUN_DEBOUNCE_MS)

    def on_debug_control_change(*_):
        if is_batch_running_for_preview():
            update_preview_runtime_mode()
            return

        for refresher in control_refreshers:
            refresher()
        session['pending_reset_scroll'] = False
        if not apply_debug_visual_from_cache(reset_scroll=False):
            schedule_preview_rerun(reason="Debug mask", delay_ms=0)

    def on_zoom_change(_event=None):
        _mode, normalized = normalize_zoom_value(zoom_var.get())
        if zoom_var.get().strip() != normalized:
            zoom_var.set(normalized)
        refresh_rendered_images(reset_scroll=False)
        return "break"

    def on_canvas_resize(_event=None):
        mode, _normalized = normalize_zoom_value(zoom_var.get())
        if mode in {'fit_width', 'fit_height'}:
            refresh_rendered_images(reset_scroll=False)

    def on_ctrl_zoom_in(_event=None):
        return change_zoom_by_step(1)

    def on_ctrl_zoom_out(_event=None):
        return change_zoom_by_step(-1)

    def apply_selected_preset_preview(_event=None):
        selected_name = preset_name_var.get().strip()
        if selected_name not in self.quality_presets:
            return

        self._apply_quality_preset_payload_to_vars(self.quality_presets[selected_name])
        for refresher in control_refreshers:
            refresher()
        sync_tuning_states()
        self._sync_open_preset_selectors(source='preview', selected_name=selected_name)
        session['pending_reset_scroll'] = False
        schedule_preview_rerun(reason=f"Preset preview: {selected_name}", delay_ms=0)

    def set_selected_preset_active(scope):
        selected_name = preset_name_var.get().strip()
        if selected_name not in self.quality_presets:
            messagebox.showerror("Preset", f"Preset '{selected_name}' does not exist.", parent=preview_window)
            return

        if not self._set_active_quality_preset(selected_name, folder_path=image_folder, scope=scope):
            messagebox.showerror("Preset", "Could not set selected preset as active for this folder.", parent=preview_window)
            return
        self._apply_quality_preset_payload_to_vars(self.quality_presets[selected_name])
        for refresher in control_refreshers:
            refresher()
        sync_tuning_states()
        session['pending_reset_scroll'] = False
        schedule_preview_rerun(reason=f"Applied preset: {selected_name}", delay_ms=0)
        self.save_settings(snapshot_active_preset=False)
        self._sync_open_preset_selectors(source='preview', selected_name=selected_name)
        status_var.set(f"Active preset set to '{selected_name}' ({scope}).")

    def revert_folder_to_default():
        if not self._set_active_quality_preset('Default', folder_path=image_folder, scope='folder'):
            messagebox.showerror("Preset", "Could not revert folder override to Default.", parent=preview_window)
            return
        preset_name_var.set('Default')
        self._apply_quality_preset_payload_to_vars(self.quality_presets['Default'])
        self.save_settings(snapshot_active_preset=False)
        self._sync_open_preset_selectors(source='preview', selected_name='Default')
        for refresher in control_refreshers:
            refresher()
        sync_tuning_states()
        schedule_preview_rerun(reason="Reverted to Default", delay_ms=0)
        status_var.set("Folder preset reverted to Default.")

    def create_new_preset_action():
        new_name = simpledialog.askstring("New Preset", "New preset name:", parent=preview_window)
        if new_name is None:
            return
        clean_name = new_name.strip()
        if not clean_name:
            messagebox.showerror("Preset", "Preset name cannot be empty.", parent=preview_window)
            return

        ok, result = self._save_current_values_as_preset(clean_name, allow_overwrite=False)
        if not ok:
            messagebox.showerror("Preset", result, parent=preview_window)
            return

        self._set_active_quality_preset(result, folder_path=image_folder, scope='folder')
        self.save_settings()
        refresh_preset_selector(result)
        self._sync_open_preset_selectors(source='preview')
        status_var.set(f"Created preset '{result}'.")

    def save_to_selected_preset_action():
        selected_name = preset_name_var.get().strip()
        if selected_name == 'Default':
            messagebox.showinfo("Preset", "Use 'New preset from current' to create a user preset from Default.", parent=preview_window)
            return

        if selected_name not in self.quality_presets:
            messagebox.showerror("Preset", f"Preset '{selected_name}' does not exist.", parent=preview_window)
            return

        if not messagebox.askyesno("Overwrite Preset", f"Overwrite preset '{selected_name}' with current values?", parent=preview_window):
            return

        ok, result = self._save_current_values_as_preset(selected_name, allow_overwrite=True)
        if not ok:
            messagebox.showerror("Preset", result, parent=preview_window)
            return

        self.save_settings()
        refresh_preset_selector(result)
        self._sync_open_preset_selectors(source='preview')
        status_var.set(f"Saved current values to '{result}'.")

    def rename_selected_preset_action():
        selected_name = preset_name_var.get().strip()
        new_name = simpledialog.askstring("Rename Preset", "New preset name:", initialvalue=selected_name, parent=preview_window)
        if new_name is None:
            return

        ok, result = self._rename_quality_preset(selected_name, new_name)
        if not ok:
            messagebox.showerror("Preset", result, parent=preview_window)
            return

        self.save_settings()
        refresh_preset_selector(result)
        self._sync_open_preset_selectors(source='preview')
        status_var.set(f"Renamed preset to '{result}'.")

    def delete_selected_preset_action():
        selected_name = preset_name_var.get().strip()
        available_presets = list(self._ordered_preset_names())
        if len(available_presets) == 1 and available_presets[0] == 'Default':
            messagebox.showinfo("Preset", "Default is the only preset available and cannot be deleted.", parent=preview_window)
            return
        if not messagebox.askyesno("Delete Preset", f"Delete preset '{selected_name}'?", parent=preview_window):
            return

        ok, _ = self._delete_quality_preset(selected_name)
        if not ok:
            messagebox.showerror("Preset", f"Could not delete '{selected_name}'.", parent=preview_window)
            return

        fallback_name = self._active_quality_preset_for_folder(image_folder)
        self.save_settings()
        refresh_preset_selector(fallback_name)
        self._sync_open_preset_selectors(source='preview')
        status_var.set(f"Deleted preset '{selected_name}'.")

    def import_preset_action():
        import_path = filedialog.askopenfilename(
            title="Import Presets",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
            parent=preview_window,
        )
        if not import_path:
            return

        try:
            merged_count = self._import_quality_presets(import_path)
        except Exception as err:
            messagebox.showerror("Preset Import", f"Failed to import presets: {err}", parent=preview_window)
            return

        self.save_settings()
        refresh_preset_selector(self._active_quality_preset_for_folder(image_folder))
        self._sync_open_preset_selectors(source='preview')
        status_var.set(f"Imported {merged_count} preset(s).")

    def export_preset_action():
        export_path = filedialog.asksaveasfilename(
            title="Export Presets",
            defaultextension=".json",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
            initialfile="colorizer-presets.json",
            parent=preview_window,
        )
        if not export_path:
            return

        try:
            self._export_quality_presets(export_path)
        except Exception as err:
            messagebox.showerror("Preset Export", f"Failed to export presets: {err}", parent=preview_window)
            return

        status_var.set(f"Exported presets to {os.path.basename(export_path)}.")

    def export_current_masks_bundle():
        source_np = session.get('last_source_np')
        luma_np = session.get('last_luminance_source_np')
        preview_colorized = session.get('last_colorized_pil')
        if source_np is None or luma_np is None or preview_colorized is None:
            messagebox.showinfo(
                "Export Masks",
                "Render a preview image first, then export masks.",
                parent=preview_window,
            )
            return

        target_dir = filedialog.askdirectory(
            title="Select folder for mask export",
            initialdir=image_folder if os.path.isdir(image_folder) else os.getcwd(),
            parent=preview_window,
        )
        if not target_dir:
            return

        image_name = os.path.splitext(os.path.basename(session.get('current_path', 'preview')))[0]

        try:
            bundle_dir = export_current_mask_bundle(
                target_dir=target_dir,
                image_name=image_name,
                source_np=source_np,
                working_np=luma_np,
                preview_colorized_pil=preview_colorized,
                active_debug_mask=debug_mask_label_to_key(debug_mask_type_var.get()),
                active_debug_mask_label=debug_mask_type_var.get(),
                active_debug_mode=debug_mask_mode_var.get(),
                active_debug_opacity=debug_mask_opacity_var.get(),
                source_image_path=session.get('current_path', ''),
            )

        except Exception as err:
            messagebox.showerror("Export Masks", f"Failed to export masks: {err}", parent=preview_window)
            return

        status_var.set(f"Exported masks bundle: {os.path.basename(bundle_dir)}")
        messagebox.showinfo("Export Masks", f"Saved mask bundle to:\n{bundle_dir}", parent=preview_window)

    def go_prev():
        set_current_index(session['current_index'] - 1, reason="Prev", immediate=True)

    def go_next():
        set_current_index(session['current_index'] + 1, reason="Next", immediate=True)

    def go_random():
        total = len(session['image_paths'])
        if total <= 1:
            set_current_index(0, reason="Random", immediate=True)
            return

        current = session['current_index']
        recent = set(list(session['history'])[-6:])
        candidates = [idx for idx in range(total) if idx != current and idx not in recent]
        if not candidates:
            candidates = [idx for idx in range(total) if idx != current]

        picked = random.choice(candidates) if candidates else current
        set_current_index(picked, reason="Random", immediate=True)

    def rerun_current():
        session['pending_reset_scroll'] = False
        schedule_preview_rerun(reason="Re-run Current", delay_ms=0)

    def on_preview_close():
        if session['scheduled_after'] is not None:
            try:
                preview_window.after_cancel(session['scheduled_after'])
            except tk.TclError:
                pass
            session['scheduled_after'] = None

        session['generation'] += 1
        for variable, trace_id in session['trace_handles']:
            try:
                variable.trace_remove('write', trace_id)
            except tk.TclError:
                pass

        session['refresh_preset_selector'] = None

        if self.preview_session is session:
            self.preview_session = None

        preview_window.destroy()

        def check_preview_timeout():
            if self.preview_session is None and not (self.processing_thread and self.processing_thread.is_alive()):
                self.free_vram()
                self.log("[*] Preview closed for 5 seconds. VRAM released.")

        self.after(5000, check_preview_timeout)

    def update_preset_action_states():
        available_presets = list(self._ordered_preset_names())
        only_default = len(available_presets) == 1 and available_presets[0] == 'Default'
        preset_action_menu.entryconfigure("Delete selected", state="disabled" if only_default else "normal")

    preset_action_menu = tk.Menu(preset_actions_button, tearoff=False)
    preset_action_menu.add_command(label="New preset from current...", command=create_new_preset_action)
    preset_action_menu.add_command(label="Save current to selected", command=save_to_selected_preset_action)
    preset_action_menu.add_command(label="Rename selected...", command=rename_selected_preset_action)
    preset_action_menu.add_command(label="Delete selected", command=delete_selected_preset_action)
    preset_action_menu.add_separator()
    preset_action_menu.add_command(label="Set selected active for folder", command=lambda: set_selected_preset_active('folder'))
    preset_action_menu.add_command(label="Set selected active global", command=lambda: set_selected_preset_active('global'))
    preset_action_menu.add_command(label="Revert folder to Default", command=revert_folder_to_default)
    preset_action_menu.add_separator()
    preset_action_menu.add_command(label="Import presets...", command=import_preset_action)
    preset_action_menu.add_command(label="Export presets...", command=export_preset_action)
    preset_action_menu.add_separator()
    preset_action_menu.add_command(label="Export current original + masks...", command=export_current_masks_bundle)
    preset_action_menu.configure(postcommand=update_preset_action_states)
    preset_actions_button.configure(menu=preset_action_menu)

    prev_button.config(command=go_prev)
    next_button.config(command=go_next)
    random_button.config(command=go_random)
    rerun_button.config(command=rerun_current)
    tuning_toggle_button.config(command=toggle_tuning)

    zoom_combo.bind("<<ComboboxSelected>>", on_zoom_change)
    zoom_combo.bind("<Return>", on_zoom_change)
    zoom_combo.bind("<FocusOut>", on_zoom_change)
    preset_selector.bind("<<ComboboxSelected>>", apply_selected_preset_preview)

    original_canvas.bind("<Configure>", on_canvas_resize, add="+")
    colorized_canvas.bind("<Configure>", on_canvas_resize, add="+")

    preview_window.bind("<Control-plus>", on_ctrl_zoom_in)
    preview_window.bind("<Control-equal>", on_ctrl_zoom_in)
    preview_window.bind("<Control-KP_Add>", on_ctrl_zoom_in)
    preview_window.bind("<Control-minus>", on_ctrl_zoom_out)
    preview_window.bind("<Control-KP_Subtract>", on_ctrl_zoom_out)

    attach_tooltip(prev_button, "Go to the previous image in the sorted folder list.")
    attach_tooltip(next_button, "Go to the next image in the sorted folder list.")
    attach_tooltip(random_button, "Pick a random image while avoiding very recent picks when possible.")
    attach_tooltip(rerun_button, "Re-run processing for the current image immediately.")
    attach_tooltip(zoom_combo, "Type percent (for example 130), use Fit Width/Fit Height, Ctrl+Wheel, or Ctrl +/- to zoom both panels.")
    attach_tooltip(preset_selector, "Choose a preset to preview instantly.")
    attach_tooltip(preset_actions_button, "Preset actions plus export of current original and debug masks.")
    attach_tooltip(tuning_toggle_button, "Show or hide compact live tuning controls. Disabled while batch processing is active.")

    live_vars = [
        self.denoise_sigma,
        self.chroma_resize_mode,
        self.edge_chroma_protection,
        self.edge_chroma_strength,
        self.line_ink_protection,
        self.line_ink_protection_strength,
        self.screentone_chroma_smoothing,
        self.screentone_smoothing_strength,
    ]
    for live_var in live_vars:
        trace_id = live_var.trace_add('write', on_live_control_change)
        session['trace_handles'].append((live_var, trace_id))

    debug_only_vars = [
        debug_mask_type_var,
        debug_mask_mode_var,
        debug_mask_opacity_var,
    ]
    for debug_var in debug_only_vars:
        trace_id = debug_var.trace_add('write', on_debug_control_change)
        session['trace_handles'].append((debug_var, trace_id))

    sync_tuning_states()
    for refresher in control_refreshers:
        refresher()

    refresh_preset_selector(active_folder_preset)
    refresh_image_caption()
    update_preview_runtime_mode()
    preview_window.protocol("WM_DELETE_WINDOW", on_preview_close)
    set_current_index(0, reason="Initial load", immediate=True)

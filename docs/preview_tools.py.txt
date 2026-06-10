"""
Professional Mask Editor — Photoshop-class selection tools for manga mask editing.

Provides brush, eraser, free lasso, polygonal lasso, rectangle marquee,
ellipse marquee, and magic wand tools with undo/redo, selection modes,
visual feedback, and keyboard shortcuts.
"""

import tkinter as tk
import numpy as np
from PIL import Image, ImageTk, ImageDraw
import cv2


# ---------------------------------------------------------------------------
# Keyboard Shortcut Map (shown in tooltips)
# ---------------------------------------------------------------------------
TOOL_SHORTCUTS = {
    'brush':           'B',
    'eraser':          'E',
    'free_lasso':      'L',
    'poly_lasso':      'P',
    'rect_marquee':    'M',
    'ellipse_marquee': 'Shift+M',
    'magic_wand':      'W',
}

TOOL_DISPLAY_NAMES = {
    'brush':           'Brush',
    'eraser':          'Eraser',
    'free_lasso':      'Free Lasso',
    'poly_lasso':      'Polygonal Lasso',
    'rect_marquee':    'Rectangle',
    'ellipse_marquee': 'Ellipse',
    'magic_wand':      'Magic Wand',
}


def tool_tooltip(tool_name):
    """Return display name with shortcut for tooltip."""
    name = TOOL_DISPLAY_NAMES.get(tool_name, tool_name)
    key = TOOL_SHORTCUTS.get(tool_name, '')
    return f"{name} ({key})" if key else name


# ---------------------------------------------------------------------------
# Undo/Redo History
# ---------------------------------------------------------------------------
class MaskHistory:
    """Circular buffer of mask snapshots for undo/redo."""

    def __init__(self, max_depth=20):
        self.stack = []
        self.position = -1
        self.max_depth = max_depth

    def push(self, mask_array):
        """Save a snapshot after each tool operation."""
        # Discard any redo history beyond current position
        self.stack = self.stack[:self.position + 1]
        self.stack.append(mask_array.copy())
        # Trim oldest if over capacity
        if len(self.stack) > self.max_depth:
            self.stack = self.stack[-self.max_depth:]
        self.position = len(self.stack) - 1

    def undo(self):
        """Return previous mask state, or None if at start."""
        if self.position > 0:
            self.position -= 1
            return self.stack[self.position].copy()
        return None

    def redo(self):
        """Return next mask state, or None if at end."""
        if self.position < len(self.stack) - 1:
            self.position += 1
            return self.stack[self.position].copy()
        return None

    def can_undo(self):
        return self.position > 0

    def can_redo(self):
        return self.position < len(self.stack) - 1

    def clear(self):
        self.stack.clear()
        self.position = -1


# ---------------------------------------------------------------------------
# Main Workspace
# ---------------------------------------------------------------------------
class MaskEditorWorkspace:
    """
    Professional mask editing workspace with Photoshop-class tools.

    Supports: brush, eraser, free lasso, polygonal lasso,
    rectangle marquee, ellipse marquee, magic wand.
    """

    def __init__(self, canvas, image_id, on_mask_changed_callback=None,
                 source_image_gray=None, on_mask_live_update=None,
                 pixel_wipe_active_check=None):
        self.canvas = canvas
        self.image_id = image_id
        self.on_mask_changed = on_mask_changed_callback
        self.on_mask_live_update = on_mask_live_update  # Called for live colorize rerun
        self._pixel_wipe_active = pixel_wipe_active_check  # Callable → bool

        # Mask state
        self.mask_pil = None       # User edits layer (additions/subtractions)
        self.mask_draw = None
        self._base_mask = None     # OCR-generated mask (read-only reference)
        self.image_width = 1
        self.image_height = 1
        self.zoom_factor = 1.0

        # Source image for magic wand
        self.source_gray = source_image_gray

        # Tool state
        self.active_tool = None
        self.selection_mode = 'add'   # add / subtract / intersect
        self.brush_size = 20
        self.wand_tolerance = 32
        self.feather_radius = 0

        # Drawing state
        self._is_drawing = False
        self._last_x = 0
        self._last_y = 0
        self._brush_color = 255      # Current stroke color
        self._brush_mode = 'add'     # add / subtract / restore / erase

        # Lasso state
        self._lasso_points = []       # Canvas coords
        self._lasso_canvas_ids = []   # Canvas item IDs for visual feedback

        # Polygonal lasso state
        self._poly_points = []        # Image coords
        self._poly_canvas_ids = []
        self._poly_rubber_band = None

        # Marquee state
        self._marquee_start = None    # Image coords (x, y)
        self._marquee_rect_id = None  # Canvas item ID

        # Visual feedback
        self._cursor_circle_id = None
        self._overlay_photo = None
        self._overlay_item_id = None
        self._stroke_canvas_ids = []   # Lightweight canvas lines for real-time brush feedback
        self._overlay_dirty = False    # Deferred overlay rendering
        self.show_mask_overlay = True
        self.mask_overlay_opacity = 0.4

        # History
        self.history = MaskHistory(max_depth=20)

        # Bind events
        self._bind_events()

    # ------------------------------------------------------------------
    # Mask Management
    # ------------------------------------------------------------------
    def reset_mask(self, width, height, zoom_factor):
        """Create a blank mask for the given image dimensions."""
        self.image_width = width
        self.image_height = height
        self.zoom_factor = zoom_factor
        self.mask_pil = Image.new('L', (width, height), 0)
        self.mask_draw = ImageDraw.Draw(self.mask_pil)
        self._base_mask = None
        self.history.clear()
        self.history.push(np.array(self.mask_pil))
        self._cancel_active_selection()

    def set_base_mask(self, mask_array):
        """
        Set the OCR-generated base mask. User edits are layered on top.

        If the user already has edits (non-empty history), only updates
        the base reference without overwriting current mask_pil.
        """
        self._base_mask = mask_array.copy()
        self.image_height, self.image_width = mask_array.shape[:2]

        # Only initialize mask_pil from OCR mask on first load
        # (empty history = no user edits yet)
        if not self.history.stack:
            self.mask_pil = Image.fromarray(mask_array).convert('L')
            self.mask_draw = ImageDraw.Draw(self.mask_pil)
            self.history.push(mask_array.copy())
        self._render_overlay()

    def reset_to_base_mask(self):
        """Reset user edits back to the OCR base mask."""
        if self._base_mask is not None:
            self.mask_pil = Image.fromarray(self._base_mask).convert('L')
            self.mask_draw = ImageDraw.Draw(self.mask_pil)
            self.history.push(self._base_mask.copy())
            self._notify_mask_changed()

    def load_mask(self, mask_array, zoom_factor):
        """Load an existing mask array."""
        self.zoom_factor = zoom_factor
        self.image_width = mask_array.shape[1]
        self.image_height = mask_array.shape[0]
        self.mask_pil = Image.fromarray(mask_array).convert('L')
        self.mask_draw = ImageDraw.Draw(self.mask_pil)
        self.history.clear()
        self.history.push(mask_array.copy())
        self._cancel_active_selection()

    def get_mask_array(self):
        """Return current mask as numpy array (includes user edits)."""
        if self.mask_pil is None:
            return None
        return np.array(self.mask_pil)

    def get_final_mask(self):
        """
        Return the final combined mask (base + user edits).
        This is what should be used for colorization.
        """
        return self.get_mask_array()

    def set_source_image(self, gray_image):
        """Set the source grayscale image (used by magic wand)."""
        self.source_gray = gray_image

    # ------------------------------------------------------------------
    # Tool Selection
    # ------------------------------------------------------------------
    def set_tool(self, tool_name):
        """Activate a tool. Pass None to deactivate all tools."""
        self._cancel_active_selection()
        self.active_tool = tool_name

        cursors = {
            'brush': 'circle',
            'eraser': 'circle',
            'free_lasso': 'crosshair',
            'poly_lasso': 'crosshair',
            'rect_marquee': 'crosshair',
            'ellipse_marquee': 'crosshair',
            'magic_wand': 'crosshair',
        }
        self.canvas.config(cursor=cursors.get(tool_name, ''))

        # Hide brush cursor when not using brush/eraser
        if tool_name not in ('brush', 'eraser'):
            self._hide_cursor_circle()

    # ------------------------------------------------------------------
    # Undo / Redo
    # ------------------------------------------------------------------
    def undo(self):
        """Undo the last mask operation."""
        state = self.history.undo()
        if state is not None:
            self.mask_pil = Image.fromarray(state).convert('L')
            self.mask_draw = ImageDraw.Draw(self.mask_pil)
            self._notify_mask_changed()
            return True
        return False

    def redo(self):
        """Redo the last undone operation."""
        state = self.history.redo()
        if state is not None:
            self.mask_pil = Image.fromarray(state).convert('L')
            self.mask_draw = ImageDraw.Draw(self.mask_pil)
            self._notify_mask_changed()
            return True
        return False

    # ------------------------------------------------------------------
    # Event Bindings
    # ------------------------------------------------------------------
    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self._on_press, add="+")
        self.canvas.bind("<B1-Motion>", self._on_motion, add="+")
        self.canvas.bind("<ButtonRelease-1>", self._on_release, add="+")
        self.canvas.bind("<Motion>", self._on_cursor_move, add="+")
        self.canvas.bind("<Double-Button-1>", self._on_double_click, add="+")
        self.canvas.bind("<Escape>", self._on_escape, add="+")
        self.canvas.bind("<ButtonPress-3>", self._on_right_click, add="+")

    def bind_keyboard_shortcuts(self, widget):
        """Bind keyboard shortcuts to a parent widget (e.g., the preview window)."""
        widget.bind('<Key-b>', lambda e: self.set_tool('brush'))
        widget.bind('<Key-e>', lambda e: self.set_tool('eraser'))
        widget.bind('<Key-l>', lambda e: self.set_tool('free_lasso'))
        widget.bind('<Key-p>', lambda e: self.set_tool('poly_lasso'))
        widget.bind('<Key-m>', lambda e: self.set_tool('rect_marquee'))
        widget.bind('<Shift-Key-M>', lambda e: self.set_tool('ellipse_marquee'))
        widget.bind('<Key-w>', lambda e: self.set_tool('magic_wand'))
        widget.bind('<Control-z>', lambda e: self.undo())
        widget.bind('<Control-Z>', lambda e: self.undo())
        widget.bind('<Control-Shift-z>', lambda e: self.redo())
        widget.bind('<Control-Shift-Z>', lambda e: self.redo())
        widget.bind('<Control-y>', lambda e: self.redo())
        widget.bind('<Control-Y>', lambda e: self.redo())
        widget.bind('<bracketleft>', lambda e: self._adjust_brush_size(-5))
        widget.bind('<bracketright>', lambda e: self._adjust_brush_size(5))
        widget.bind('<Key-v>', lambda e: self.toggle_overlay())
        widget.bind('<Key-r>', lambda e: self.reset_to_base_mask())
        widget.bind('<Return>', lambda e: self._on_enter_key())
        widget.bind('<KP_Enter>', lambda e: self._on_enter_key())

    # ------------------------------------------------------------------
    # Coordinate Conversion
    # ------------------------------------------------------------------
    def _canvas_to_image(self, cx, cy):
        """Convert canvas widget coords to image pixel coords."""
        canvas_x = self.canvas.canvasx(cx)
        canvas_y = self.canvas.canvasy(cy)
        img_x = int(canvas_x / self.zoom_factor)
        img_y = int(canvas_y / self.zoom_factor)
        return max(0, min(img_x, self.image_width - 1)), \
               max(0, min(img_y, self.image_height - 1))

    def _image_to_canvas(self, ix, iy):
        """Convert image pixel coords to canvas coords."""
        return ix * self.zoom_factor, iy * self.zoom_factor

    # ------------------------------------------------------------------
    # Mouse Events
    # ------------------------------------------------------------------
    def _on_press(self, event):
        if not self.active_tool or self.mask_pil is None:
            return
        # Skip if pixel-wipe mode is active (avoid event conflict)
        if self._pixel_wipe_active and self._pixel_wipe_active():
            return

        # Determine selection mode from modifier keys
        shift = bool(event.state & 0x0001)
        alt = bool(event.state & 0x20000) or bool(event.state & 0x0008)
        if shift and alt:
            self.selection_mode = 'intersect'
        elif alt:
            self.selection_mode = 'subtract'
        elif shift:
            self.selection_mode = 'subtract'  # Shift = remove from mask (FP cleanup)
        else:
            self.selection_mode = 'add'

        img_x, img_y = self._canvas_to_image(event.x, event.y)

        if self.active_tool in ('brush', 'eraser'):
            self._is_drawing = True
            self._last_x, self._last_y = img_x, img_y
            self._last_canvas_x = self.canvas.canvasx(event.x)
            self._last_canvas_y = self.canvas.canvasy(event.y)

            # Modifier-aware color:
            # Brush: normal=add(255), shift=subtract(0)
            # Eraser: normal=erase(0), shift=restore base, alt=erase(0)
            if self.active_tool == 'brush':
                if shift:
                    self._brush_color = 0
                    self._brush_mode = 'subtract'
                else:
                    self._brush_color = 255
                    self._brush_mode = 'add'
            else:  # eraser
                if shift:
                    self._brush_color = None
                    self._brush_mode = 'restore'
                elif alt:
                    self._brush_color = 0
                    self._brush_mode = 'subtract'
                else:
                    self._brush_color = 0
                    self._brush_mode = 'erase'

            r = self.brush_size // 2
            if self._brush_mode == 'restore':
                self._apply_restore_dot(img_x, img_y, r)
            else:
                color = self._brush_color if self._brush_color is not None else 0
                self.mask_draw.ellipse(
                    [img_x - r, img_y - r, img_x + r, img_y + r],
                    fill=color
                )

            # Lightweight canvas dot for real-time feedback
            canvas_r = max(1, int(r * self.zoom_factor))
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            if self._brush_mode == 'restore':
                stroke_color = '#33FF66'
            elif self._brush_mode in ('subtract', 'erase'):
                stroke_color = '#333333'
            else:
                stroke_color = '#3388FF'
            dot_id = self.canvas.create_oval(
                cx - canvas_r, cy - canvas_r,
                cx + canvas_r, cy + canvas_r,
                fill=stroke_color, outline='', stipple='gray50'
            )
            self._stroke_canvas_ids.append(dot_id)

        elif self.active_tool == 'free_lasso':
            self._is_drawing = True
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            self._lasso_points = [(cx, cy)]

        elif self.active_tool == 'poly_lasso':
            # Add point; first click starts, subsequent clicks add vertices
            self._poly_points.append((img_x, img_y))
            cx, cy = self._image_to_canvas(img_x, img_y)
            if len(self._poly_points) > 1:
                prev_cx, prev_cy = self._image_to_canvas(
                    *self._poly_points[-2]
                )
                line_id = self.canvas.create_line(
                    prev_cx, prev_cy, cx, cy,
                    fill='#FF3366', width=2, dash=(6, 3)
                )
                self._poly_canvas_ids.append(line_id)
            # Draw vertex dot
            dot_id = self.canvas.create_oval(
                cx - 3, cy - 3, cx + 3, cy + 3,
                fill='#FF3366', outline='white'
            )
            self._poly_canvas_ids.append(dot_id)

        elif self.active_tool in ('rect_marquee', 'ellipse_marquee'):
            self._is_drawing = True
            self._marquee_start = (img_x, img_y)

        elif self.active_tool == 'magic_wand':
            self._apply_magic_wand(img_x, img_y)

    def _on_motion(self, event):
        if not self.active_tool or self.mask_pil is None:
            return

        img_x, img_y = self._canvas_to_image(event.x, event.y)

        if self.active_tool in ('brush', 'eraser') and self._is_drawing:
            if self._brush_mode == 'restore':
                self._apply_restore_line(
                    self._last_x, self._last_y, img_x, img_y
                )
            else:
                color = self._brush_color if self._brush_color is not None else 0
                self.mask_draw.line(
                    [(self._last_x, self._last_y), (img_x, img_y)],
                    fill=color, width=self.brush_size, joint='curve'
                )
            # Lightweight canvas line for real-time feedback
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            if self._brush_mode == 'restore':
                stroke_color = '#33FF66'
            elif self._brush_mode in ('subtract', 'erase'):
                stroke_color = '#333333'
            else:
                stroke_color = '#3388FF'
            canvas_width = max(1, int(self.brush_size * self.zoom_factor))
            line_id = self.canvas.create_line(
                self._last_canvas_x, self._last_canvas_y, cx, cy,
                fill=stroke_color, width=canvas_width,
                capstyle='round', joinstyle='round',
                stipple='gray50'
            )
            self._stroke_canvas_ids.append(line_id)
            self._last_x, self._last_y = img_x, img_y
            self._last_canvas_x, self._last_canvas_y = cx, cy

        elif self.active_tool == 'free_lasso' and self._is_drawing:
            cx = self.canvas.canvasx(event.x)
            cy = self.canvas.canvasy(event.y)
            self._lasso_points.append((cx, cy))
            # Draw trail line segment
            if len(self._lasso_points) >= 2:
                p1 = self._lasso_points[-2]
                p2 = self._lasso_points[-1]
                line_id = self.canvas.create_line(
                    p1[0], p1[1], p2[0], p2[1],
                    fill='#FF3366', width=2, dash=(4, 4)
                )
                self._lasso_canvas_ids.append(line_id)

        elif self.active_tool == 'poly_lasso' and self._poly_points:
            # Rubber band line from last vertex to cursor
            last_ix, last_iy = self._poly_points[-1]
            last_cx, last_cy = self._image_to_canvas(last_ix, last_iy)
            cur_cx = self.canvas.canvasx(event.x)
            cur_cy = self.canvas.canvasy(event.y)
            if self._poly_rubber_band:
                self.canvas.coords(
                    self._poly_rubber_band,
                    last_cx, last_cy, cur_cx, cur_cy
                )
            else:
                self._poly_rubber_band = self.canvas.create_line(
                    last_cx, last_cy, cur_cx, cur_cy,
                    fill='#FF3366', width=1, dash=(3, 3)
                )

        elif self.active_tool in ('rect_marquee', 'ellipse_marquee') and self._is_drawing:
            self._draw_marquee_preview(img_x, img_y)

    def _on_release(self, event):
        if not self.active_tool or self.mask_pil is None:
            return

        if self.active_tool in ('brush', 'eraser') and self._is_drawing:
            self._is_drawing = False
            # Clean up lightweight stroke feedback
            for sid in self._stroke_canvas_ids:
                self.canvas.delete(sid)
            self._stroke_canvas_ids.clear()
            # Now do the expensive overlay render + callback (once)
            self.history.push(np.array(self.mask_pil))
            self._notify_mask_changed()

        elif self.active_tool == 'free_lasso' and self._is_drawing:
            self._is_drawing = False
            self._finalize_free_lasso()

        elif self.active_tool in ('rect_marquee', 'ellipse_marquee') and self._is_drawing:
            self._is_drawing = False
            img_x, img_y = self._canvas_to_image(event.x, event.y)
            self._finalize_marquee(img_x, img_y)

    def _on_double_click(self, event):
        """Double-click closes polygonal lasso."""
        if self.active_tool == 'poly_lasso' and len(self._poly_points) > 2:
            # Remove last point if too close to previous (double-click artifact)
            if len(self._poly_points) >= 2:
                p1 = self._poly_points[-1]
                p2 = self._poly_points[-2]
                if abs(p1[0]-p2[0]) < 5 and abs(p1[1]-p2[1]) < 5:
                    self._poly_points.pop()
            if len(self._poly_points) > 2:
                self._finalize_poly_lasso()

    def _on_enter_key(self):
        """Enter key closes polygonal lasso."""
        if self.active_tool == 'poly_lasso' and len(self._poly_points) > 2:
            self._finalize_poly_lasso()

    def _on_escape(self, event):
        """Escape cancels current selection in progress."""
        self._cancel_active_selection()

    def _on_right_click(self, event):
        """Right-click removes last vertex in polygonal lasso."""
        if self.active_tool == 'poly_lasso' and self._poly_points:
            self._poly_points.pop()
            if self._poly_canvas_ids:
                # Remove last 2 canvas items (line + dot)
                for _ in range(min(2, len(self._poly_canvas_ids))):
                    cid = self._poly_canvas_ids.pop()
                    self.canvas.delete(cid)

    def _on_cursor_move(self, event):
        """Update brush cursor circle on mouse move."""
        if self.active_tool in ('brush', 'eraser'):
            self._draw_cursor_circle(event.x, event.y)

    # ------------------------------------------------------------------
    # Tool Finalization
    # ------------------------------------------------------------------
    def _finalize_free_lasso(self):
        """Close and fill the free lasso selection."""
        # Draw closing line from last to first for visual confirmation
        if len(self._lasso_points) >= 3:
            p1 = self._lasso_points[-1]
            p0 = self._lasso_points[0]
            close_id = self.canvas.create_line(
                p1[0], p1[1], p0[0], p0[1],
                fill='#33FF66', width=2
            )
            self._lasso_canvas_ids.append(close_id)

        # Clean up canvas lines after brief delay
        canvas_ids = list(self._lasso_canvas_ids)
        self._lasso_canvas_ids.clear()

        if len(self._lasso_points) < 3:
            for cid in canvas_ids:
                self.canvas.delete(cid)
            self._lasso_points.clear()
            return

        # Convert canvas coords to image coords
        img_points = [
            (int(pt[0] / self.zoom_factor),
             int(pt[1] / self.zoom_factor))
            for pt in self._lasso_points
        ]
        self._apply_polygon_to_mask(img_points)
        self._lasso_points.clear()

        # Flash green then clean up
        self.canvas.after(200, lambda: [
            self.canvas.delete(cid) for cid in canvas_ids
        ])

    def _finalize_poly_lasso(self):
        """Close and fill the polygonal lasso selection."""
        # Clean up canvas items
        for cid in self._poly_canvas_ids:
            self.canvas.delete(cid)
        self._poly_canvas_ids.clear()
        if self._poly_rubber_band:
            self.canvas.delete(self._poly_rubber_band)
            self._poly_rubber_band = None

        if len(self._poly_points) < 3:
            self._poly_points.clear()
            return

        self._apply_polygon_to_mask(self._poly_points)
        self._poly_points.clear()

    def _finalize_marquee(self, end_x, end_y):
        """Fill the rectangle or ellipse marquee selection."""
        # Clean up preview
        if self._marquee_rect_id:
            self.canvas.delete(self._marquee_rect_id)
            self._marquee_rect_id = None

        if self._marquee_start is None:
            return

        sx, sy = self._marquee_start
        ex, ey = end_x, end_y
        self._marquee_start = None

        x1, x2 = min(sx, ex), max(sx, ex)
        y1, y2 = min(sy, ey), max(sy, ey)

        if x2 - x1 < 2 or y2 - y1 < 2:
            return

        # Green flash confirmation
        c_x1, c_y1 = self._image_to_canvas(x1, y1)
        c_x2, c_y2 = self._image_to_canvas(x2, y2)
        if self.active_tool == 'rect_marquee':
            flash_id = self.canvas.create_rectangle(
                c_x1, c_y1, c_x2, c_y2,
                fill='#33FF66', outline='', stipple='gray50'
            )
        else:
            flash_id = self.canvas.create_oval(
                c_x1, c_y1, c_x2, c_y2,
                fill='#33FF66', outline='', stipple='gray50'
            )
        self.canvas.after(200, lambda: self.canvas.delete(flash_id))

        # Create selection mask
        sel_mask = np.zeros(
            (self.image_height, self.image_width), dtype=np.uint8
        )
        if self.active_tool == 'rect_marquee':
            cv2.rectangle(sel_mask, (x1, y1), (x2, y2), 255, -1)
        else:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            rx = (x2 - x1) // 2
            ry = (y2 - y1) // 2
            cv2.ellipse(sel_mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)

        self._apply_selection_mask(sel_mask)

    def _apply_magic_wand(self, seed_x, seed_y):
        """Flood-fill selection by brightness similarity."""
        if self.source_gray is None:
            return

        src = self.source_gray
        if src.shape[:2] != (self.image_height, self.image_width):
            return

        h, w = src.shape[:2]
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        tol = self.wand_tolerance

        cv2.floodFill(
            src.copy(), flood_mask,
            (seed_x, seed_y),
            newVal=255,
            loDiff=(tol,), upDiff=(tol,),
            flags=cv2.FLOODFILL_MASK_ONLY | (255 << 8)
        )

        sel_mask = flood_mask[1:-1, 1:-1]
        self._apply_selection_mask(sel_mask)

    # ------------------------------------------------------------------
    # Selection Application
    # ------------------------------------------------------------------
    def _apply_polygon_to_mask(self, points):
        """Apply a polygon selection to the mask using current mode."""
        sel_mask = np.zeros(
            (self.image_height, self.image_width), dtype=np.uint8
        )
        pts = np.array(points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(sel_mask, [pts], 255)
        self._apply_selection_mask(sel_mask)

    def _apply_selection_mask(self, sel_mask):
        """
        Apply a binary selection mask to the working mask
        using the current selection mode.
        """
        # Optional feathering
        if self.feather_radius > 0:
            ksize = self.feather_radius * 2 + 1
            sel_mask = cv2.GaussianBlur(sel_mask, (ksize, ksize), 0)

        current = np.array(self.mask_pil)

        if self.selection_mode == 'add':
            result = cv2.max(current, sel_mask)
        elif self.selection_mode == 'subtract':
            inv_sel = cv2.bitwise_not(sel_mask)
            result = cv2.bitwise_and(current, inv_sel)
        elif self.selection_mode == 'intersect':
            result = cv2.bitwise_and(current, sel_mask)
        else:
            result = cv2.max(current, sel_mask)

        self.mask_pil = Image.fromarray(result).convert('L')
        self.mask_draw = ImageDraw.Draw(self.mask_pil)
        self.history.push(result)
        self._notify_mask_changed()

    # ------------------------------------------------------------------
    # Visual Feedback
    # ------------------------------------------------------------------
    def _draw_cursor_circle(self, wx, wy):
        """Draw a circle showing current brush size at cursor position."""
        if self._cursor_circle_id:
            self.canvas.delete(self._cursor_circle_id)

        cx = self.canvas.canvasx(wx)
        cy = self.canvas.canvasy(wy)
        r = (self.brush_size / 2) * self.zoom_factor
        self._cursor_circle_id = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline='#FF3366', width=2, dash=(4, 2)
        )

    def _hide_cursor_circle(self):
        if self._cursor_circle_id:
            self.canvas.delete(self._cursor_circle_id)
            self._cursor_circle_id = None

    def _draw_marquee_preview(self, cur_x, cur_y):
        """Draw rectangle/ellipse preview while dragging."""
        if self._marquee_rect_id:
            self.canvas.delete(self._marquee_rect_id)

        if self._marquee_start is None:
            return

        sx, sy = self._marquee_start
        # Convert to canvas coords
        c_sx, c_sy = self._image_to_canvas(sx, sy)
        c_ex, c_ey = self._image_to_canvas(cur_x, cur_y)

        if self.active_tool == 'rect_marquee':
            self._marquee_rect_id = self.canvas.create_rectangle(
                c_sx, c_sy, c_ex, c_ey,
                outline='#FF3366', width=2, dash=(6, 3)
            )
        else:
            self._marquee_rect_id = self.canvas.create_oval(
                c_sx, c_sy, c_ex, c_ey,
                outline='#FF3366', width=2, dash=(6, 3)
            )

    def render_mask_overlay(self, base_photo_image):
        """
        Create a composite image showing the mask as a semi-transparent
        red overlay on top of the base image. Returns a PhotoImage.

        Parameters
        ----------
        base_photo_image : PIL.Image.Image
            The base image (RGB) to overlay the mask on.
        """
        if not self.show_mask_overlay or self.mask_pil is None:
            return None

        mask_arr = np.array(self.mask_pil)
        if not np.any(mask_arr > 0):
            return None

        # Resize mask to match display size
        disp_w, disp_h = base_photo_image.size
        if (mask_arr.shape[1], mask_arr.shape[0]) != (disp_w, disp_h):
            mask_resized = cv2.resize(
                mask_arr, (disp_w, disp_h),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            mask_resized = mask_arr

        # Create RGBA overlay
        overlay = np.zeros((disp_h, disp_w, 4), dtype=np.uint8)
        alpha_val = int(255 * self.mask_overlay_opacity)
        overlay[mask_resized > 127] = [50, 120, 255, alpha_val]

        overlay_img = Image.fromarray(overlay, 'RGBA')
        base_rgba = base_photo_image.convert('RGBA')
        composited = Image.alpha_composite(base_rgba, overlay_img)
        return composited.convert('RGB')

    # ------------------------------------------------------------------
    # Restore-from-Base Helpers (Shift+Eraser)
    # ------------------------------------------------------------------
    def _apply_restore_dot(self, cx, cy, r):
        """Restore base mask values in a circular area around (cx, cy)."""
        if self._base_mask is None:
            return
        current = np.array(self.mask_pil)
        h, w = current.shape
        y1, y2 = max(0, cy - r), min(h, cy + r + 1)
        x1, x2 = max(0, cx - r), min(w, cx + r + 1)
        if y2 <= y1 or x2 <= x1:
            return
        ys = np.arange(y1, y2) - cy
        xs = np.arange(x1, x2) - cx
        yy, xx = np.meshgrid(ys, xs, indexing='ij')
        circle = (xx * xx + yy * yy) <= r * r
        current[y1:y2, x1:x2][circle] = self._base_mask[y1:y2, x1:x2][circle]
        self.mask_pil = Image.fromarray(current).convert('L')
        self.mask_draw = ImageDraw.Draw(self.mask_pil)

    def _apply_restore_line(self, x0, y0, x1, y1):
        """Restore base mask values along a line from (x0,y0) to (x1,y1)."""
        if self._base_mask is None:
            return
        stamp = Image.new('L', (self.image_width, self.image_height), 0)
        stamp_draw = ImageDraw.Draw(stamp)
        stamp_draw.line(
            [(x0, y0), (x1, y1)],
            fill=255, width=self.brush_size, joint='curve'
        )
        stamp_arr = np.array(stamp)
        current = np.array(self.mask_pil)
        restore = stamp_arr > 127
        current[restore] = self._base_mask[restore]
        self.mask_pil = Image.fromarray(current).convert('L')
        self.mask_draw = ImageDraw.Draw(self.mask_pil)

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------
    def _cancel_active_selection(self):
        """Cancel any in-progress selection tool."""
        self._is_drawing = False

        # Free lasso cleanup
        for cid in self._lasso_canvas_ids:
            self.canvas.delete(cid)
        self._lasso_canvas_ids.clear()
        self._lasso_points.clear()

        # Poly lasso cleanup
        for cid in self._poly_canvas_ids:
            self.canvas.delete(cid)
        self._poly_canvas_ids.clear()
        if self._poly_rubber_band:
            self.canvas.delete(self._poly_rubber_band)
            self._poly_rubber_band = None
        self._poly_points.clear()

        # Marquee cleanup
        if self._marquee_rect_id:
            self.canvas.delete(self._marquee_rect_id)
            self._marquee_rect_id = None
        self._marquee_start = None

        # Cursor cleanup
        self._hide_cursor_circle()

    def _adjust_brush_size(self, delta):
        """Adjust brush size by delta, clamped to [1, 200]."""
        self.brush_size = max(1, min(200, self.brush_size + delta))
        return self.brush_size

    def _notify_mask_changed(self):
        """Notify the parent that the mask has changed and update overlay."""
        self._render_overlay()
        if self.on_mask_changed:
            self.on_mask_changed()
        if self.on_mask_live_update:
            self.on_mask_live_update()

    def toggle_overlay(self):
        """Toggle mask overlay visibility."""
        self.show_mask_overlay = not self.show_mask_overlay
        if self.show_mask_overlay:
            self._render_overlay()
        else:
            self.hide_overlay()

    def _render_overlay(self):
        """Render the mask as a semi-transparent red overlay on the canvas."""
        if self.mask_pil is None or not self.show_mask_overlay:
            # Hide overlay if it exists
            if self._overlay_item_id is not None:
                self.canvas.delete(self._overlay_item_id)
                self._overlay_item_id = None
                self._overlay_photo = None
            return

        mask_arr = np.array(self.mask_pil)
        if not np.any(mask_arr):
            # No mask content — hide overlay
            if self._overlay_item_id is not None:
                self.canvas.delete(self._overlay_item_id)
                self._overlay_item_id = None
                self._overlay_photo = None
            return

        # Create RGBA overlay: blue tint where mask > 0
        h, w = mask_arr.shape
        overlay = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        overlay_arr = np.array(overlay)

        # Blue tint with configurable opacity (standard mask convention)
        alpha = int(255 * self.mask_overlay_opacity)
        mask_bool = mask_arr > 127
        overlay_arr[mask_bool, 0] = 50    # R
        overlay_arr[mask_bool, 1] = 120   # G
        overlay_arr[mask_bool, 2] = 255   # B
        overlay_arr[mask_bool, 3] = alpha # A

        overlay = Image.fromarray(overlay_arr, 'RGBA')

        # Resize to match current zoom
        if abs(self.zoom_factor - 1.0) > 1e-6:
            disp_w = max(1, int(round(w * self.zoom_factor)))
            disp_h = max(1, int(round(h * self.zoom_factor)))
            overlay = overlay.resize((disp_w, disp_h), Image.Resampling.NEAREST)

        self._overlay_photo = ImageTk.PhotoImage(overlay)

        if self._overlay_item_id is not None:
            self.canvas.itemconfigure(self._overlay_item_id, image=self._overlay_photo)
        else:
            self._overlay_item_id = self.canvas.create_image(
                0, 0, anchor='nw', image=self._overlay_photo
            )
        # Ensure overlay is above the base image but below cursor circle
        self.canvas.tag_raise(self._overlay_item_id)
        if self._cursor_circle_id:
            self.canvas.tag_raise(self._cursor_circle_id)

    def update_overlay_zoom(self, zoom_factor):
        """Update zoom and re-render the overlay."""
        self.zoom_factor = zoom_factor
        self._render_overlay()

    def hide_overlay(self):
        """Remove the overlay from the canvas."""
        if self._overlay_item_id is not None:
            self.canvas.delete(self._overlay_item_id)
            self._overlay_item_id = None
            self._overlay_photo = None

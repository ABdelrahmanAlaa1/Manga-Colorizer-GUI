import tkinter as tk

class HoverTooltip:
    def __init__(self, widget, text, delay_ms=450):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip_window = None

        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event=None):
        self._cancel_schedule()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, _event=None):
        self._cancel_schedule()
        self._hide()

    def _cancel_schedule(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        if self._tip_window is not None or not self.text:
            return

        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        except tk.TclError:
            return

        self._tip_window = tk.Toplevel(self.widget)
        self._tip_window.wm_overrideredirect(True)
        self._tip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self._tip_window,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            background="#fffbe6",
            padx=6,
            pady=4,
            wraplength=360,
        )
        label.pack()

    def _hide(self):
        if self._tip_window is not None:
            try:
                self._tip_window.destroy()
            except tk.TclError:
                pass
            self._tip_window = None

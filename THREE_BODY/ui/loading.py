import tkinter as tk
from tkinter import ttk

class LoadingUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("⏳ Đang tải...")
        self.root.geometry("400x180")
        self.root.configure(bg="#2c3e50")
        self.root.resizable(False, False)

        self.progress_var = tk.DoubleVar()

        self._setup_style()
        self._setup_widgets()

    def _setup_style(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor='#34495e',
                        background='#1abc9c',
                        thickness=20,
                        bordercolor='#2c3e50',
                        relief='flat')

    def _setup_widgets(self):
        tk.Label(self.root,
                 text="⏳ Đang xử lý, vui lòng chờ...",
                 font=("Segoe UI", 12, "bold"),
                 bg="#2c3e50", fg="white").pack(pady=(30, 15))

        self.progress_bar = ttk.Progressbar(
            self.root,
            length=300,
            mode='determinate',
            variable=self.progress_var,
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(pady=10)

        self.percent_label = tk.Label(
            self.root, text="0%",
            font=("Segoe UI", 10),
            bg="#2c3e50", fg="white"
        )
        self.percent_label.pack()

    def update_progress(self, percent):
        """Cập nhật tiến trình trực tiếp, gọi từ main thread."""
        self.progress_var.set(percent)
        self.percent_label.config(text=f"{int(percent)}%")

    def run(self):
        self.root.mainloop()

    def destroy(self):
        self.root.destroy()

    def close(self):
        self.root.withdraw()

    def show(self):
        self.root.deiconify()

    def close_after_delay(self, delay_ms=1000):
        self.root.after(delay_ms, self.destroy)

    def get_root(self):
        return self.root

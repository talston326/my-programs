# Interactive Line Fitter (Tkinter + Matplotlib)
# ------------------------------------------------
# How to run:
#   pip install matplotlib numpy
#   python interactive_line_fitter.py
#
# Click on the graph to add points. Points are listed at right.
# Adjust slope (m) and intercept (b), or press "Fit Line (Least Squares)".

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class LineFitterApp:
    def __init__(self, master):
        self.master = master
        master.title("Interactive Line Fitter")

        # Main layout: left = plot, right = controls
        self.left = ttk.Frame(master, padding=6)
        self.right = ttk.Frame(master, padding=6)
        self.left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right.pack(side=tk.RIGHT, fill=tk.Y)

        # Matplotlib Figure
        self.fig = Figure(figsize=(6, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)

        # Axes setup: -10..10 with integer ticks, grid, and bold axes lines
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xticks(np.arange(-10, 11, 1))
        self.ax.set_yticks(np.arange(-10, 11, 1))
        self.ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.5)
        self.ax.axhline(0, linewidth=1.5, color='black')
        self.ax.axvline(0, linewidth=1.5, color='black')
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        # Artists: scatter for points, line for y = m x + b
        self.points = []  # list of (x, y)
        self.scatter = self.ax.scatter([], [], s=40, zorder=3)
        self.line_m = 1.0
        self.line_b = 0.0
        xs = np.linspace(-10, 10, 400)
        ys = self.line_m * xs + self.line_b
        self.line_plot, = self.ax.plot(xs, ys, linewidth=2, zorder=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.left)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Connect click handler
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # ---- Right panel widgets ----
        # Instructions
        ttk.Label(self.right, text="Click on graph to add a point.").pack(anchor='w')

        # Snap-to-grid option
        self.snap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.right, text="Snap to integer grid", variable=self.snap_var).pack(anchor='w', pady=(0,8))

        # Points List
        ttk.Label(self.right, text="Points (x, y):").pack(anchor='w')
        self.points_list = tk.Listbox(self.right, height=12, width=18, activestyle='dotbox')
        self.points_list.pack(fill=tk.Y, padx=0, pady=(2,6))

        btns_frame = ttk.Frame(self.right)
        btns_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Button(btns_frame, text="Undo Last", command=self.undo_last).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,4))
        ttk.Button(btns_frame, text="Delete Selected", command=self.delete_selected).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4,4))
        ttk.Button(btns_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4,0))

        # Controls for m and b
        controls = ttk.LabelFrame(self.right, text="Line: y = m·x + b")
        controls.pack(fill=tk.X, pady=(4,8))

        self.m_var = tk.DoubleVar(value=1.0)
        self.b_var = tk.DoubleVar(value=0.0)

        row = ttk.Frame(controls)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="m (slope):").pack(side=tk.LEFT)
        self.m_spin = tk.Spinbox(row, from_=-20.0, to=20.0, increment=0.1, textvariable=self.m_var, width=8)
        self.m_spin.pack(side=tk.RIGHT)

        row2 = ttk.Frame(controls)
        row2.pack(fill=tk.X, pady=4)
        ttk.Label(row2, text="b (intercept):").pack(side=tk.LEFT)
        self.b_spin = tk.Spinbox(row2, from_=-20.0, to=20.0, increment=0.1, textvariable=self.b_var, width=8)
        self.b_spin.pack(side=tk.RIGHT)

        row3 = ttk.Frame(controls)
        row3.pack(fill=tk.X, pady=6)
        ttk.Button(row3, text="Update Line", command=self.update_line_from_vars).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,4))
        ttk.Button(row3, text="Fit Line (Least Squares)", command=self.fit_line).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4,0))

        # Equation display
        self.eq_label = ttk.Label(self.right, text=self.format_eq(self.line_m, self.line_b))
        self.eq_label.pack(anchor='w', pady=(6,4))

        # Loss display
        self.loss_label = ttk.Label(self.right, text="Loss: N/A")
        self.loss_label.pack(anchor='w', pady=(0, 4))

        # Reset view button
        ttk.Button(self.right, text="Reset View", command=self.reset_view).pack(anchor='w', pady=(2,0))

        # Bind Enter key to update the line
        master.bind('<Return>', lambda e: self.update_line_from_vars())

    # ---- Helper methods ----
    def format_eq(self, m, b):
        m_str = f"{m:.3f}"
        b_str = f"{b:.3f}"
        sign = "+" if b >= 0 else "-"
        return f"y = {m_str}·x {sign} {abs(b):.3f}"

    def compute_loss(self):
        if not self.points:
            return None
        arr = np.array(self.points)
        x = arr[:, 0]
        y = arr[:, 1]
        y_pred = self.line_m * x + self.line_b
        errors = y - y_pred
        return np.mean(errors ** 2) / 2  # Mean Squared Error (with 1/2 factor)

    def redraw_scatter(self):
        if self.points:
            arr = np.array(self.points)
            self.scatter.set_offsets(arr)
        else:
            self.scatter.set_offsets(np.empty((0, 2)))
        self.canvas.draw_idle()

    def update_line_plot(self):
        xs = np.linspace(-10, 10, 400)
        ys = self.line_m * xs + self.line_b
        self.line_plot.set_data(xs, ys)

        # Update equation label
        self.eq_label.config(text=self.format_eq(self.line_m, self.line_b))

        # Update loss label
        loss_val = self.compute_loss()
        if loss_val is not None:
            self.loss_label.config(text=f"Loss: {loss_val:.3f}")
        else:
            self.loss_label.config(text="Loss: N/A")

        # Redraw canvas
        self.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
        if self.snap_var.get():
            x = int(round(x))
            y = int(round(y))
        # Keep within bounds
        if x < -10 or x > 10 or y < -10 or y > 10:
            return
        self.points.append((x, y))
        self.points_list.insert(tk.END, f"({x:.2f}, {y:.2f})")
        self.redraw_scatter()

    def undo_last(self):
        if not self.points:
            return
        self.points.pop()
        self.points_list.delete(tk.END)
        self.redraw_scatter()

    def delete_selected(self):
        sel = list(self.points_list.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            del self.points[idx]
            self.points_list.delete(idx)
        self.redraw_scatter()

    def clear_all(self):
        if not self.points:
            return
        if messagebox.askyesno("Clear All", "Remove all points?"):
            self.points.clear()
            self.points_list.delete(0, tk.END)
            self.redraw_scatter()

    def update_line_from_vars(self):
        try:
            self.line_m = float(self.m_var.get())
            self.line_b = float(self.b_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric values for m and b.")
            return
        self.update_line_plot()

    def fit_line(self):
        if len(self.points) < 2:
            messagebox.showinfo("Need More Points", "Add at least two points to fit a line.")
            return
        arr = np.array(self.points)
        x = arr[:, 0]
        y = arr[:, 1]
        m, b = np.polyfit(x, y, 1)
        self.line_m = float(m)
        self.line_b = float(b)
        self.m_var.set(self.line_m)
        self.b_var.set(self.line_b)
        self.update_line_plot()

    def reset_view(self):
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    try:
        root.call("ttk::style", "theme", "use", "clam")
    except Exception:
        pass
    app = LineFitterApp(root)
    root.mainloop()

import matplotlib.pyplot as plt
import tkinter as tk
print(tk.TkVersion)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class VisualizationApp:
    def __init__(self, plots, titles):
        self.root = tk.Tk()
        self.root.title("Adatvizualizáció")
        
        self.plots = plots
        self.titles = titles
        self.current_plot = 0
        
        # Frame az irányításhoz
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Gombok
        self.prev_button = tk.Button(self.control_frame, text="Előző", command=self.show_previous)
        self.prev_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.next_button = tk.Button(self.control_frame, text="Következő", command=self.show_next)
        self.next_button.pack(side=tk.RIGHT, padx=5, pady=5)
        
        # Matplotlib Figure és Canvas
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Első grafikon megjelenítése
        self.show_plot(0)
        
    def show_plot(self, index):
        """Mutassa a megadott indexű grafikont."""
        self.ax.clear()
        self.plots[index](self.ax)
        self.ax.set_title(self.titles[index])
        self.canvas.draw()
        
    def show_previous(self):
        """Előző grafikon megjelenítése."""
        if self.current_plot > 0:
            self.current_plot -= 1
            self.show_plot(self.current_plot)
        
    def show_next(self):
        """Következő grafikon megjelenítése."""
        if self.current_plot < len(self.plots) - 1:
            self.current_plot += 1
            self.show_plot(self.current_plot)
        
    def run(self):
        """Indítsa el az alkalmazást."""
        self.root.mainloop()
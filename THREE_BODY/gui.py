import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from states import named_states, generate_random_state, generate_random_state_sym

def generate_random_state(dim):
    size = 12 if dim == "2D" else 18
    return np.random.uniform(-1, 1, size)

def generate_random_state_sym(dim):
    size = 12 if dim == "2D" else 18
    state = np.random.uniform(-1, 1, size)
    return state

class ModernUI:
    """Base class for modern UI elements"""
    def __init__(self):
        # Modern color scheme
        self.colors = {
            "primary": "#4361ee",
            "secondary": "#3f37c9",
            "accent": "#4895ef",
            "success": "#4cc9f0",
            "warning": "#f72585",
            "light_bg": "#f8f9fa",
            "dark_bg": "#212529",
            "text": "#212529",
            "light_text": "#f8f9fa",
            "border": "#dee2e6",
            "hover": "#e9ecef"
        }
        
        # Font configurations
        self.fonts = {
            "heading": ("Helvetica", 16, "bold"),
            "subheading": ("Helvetica", 12, "bold"),
            "body": ("Helvetica", 10),
            "small": ("Helvetica", 8),
            "button": ("Helvetica", 10, "bold")
        }

    def create_modern_button(self, parent, text, command, bg_color=None, fg_color=None, width=None, height=None):
        """Create a modern-looking button"""
        if bg_color is None:
            bg_color = self.colors["primary"]
        if fg_color is None:
            fg_color = self.colors["light_text"]
            
        button = tk.Button(
            parent, 
            text=text,
            command=command,
            bg=bg_color,
            fg=fg_color,
            relief=tk.FLAT,
            font=self.fonts["button"],
            activebackground=self.colors["secondary"],
            activeforeground=self.colors["light_text"],
            padx=10,
            pady=5,
            width=width,
            height=height,
            cursor="hand2"
        )
        
        # Hover effect
        def on_enter(e):
            button['background'] = self.colors["secondary"]
        
        def on_leave(e):
            button['background'] = bg_color
            
        button.bind("<Enter>", on_enter)
        button.bind("<Leave>", on_leave)
        
        return button

class StateSelector(ModernUI):
    def __init__(self):
        super().__init__()
        self.selected_state = {"state": None, "dim": "2D"}
        self.entries = []
        self.setup_ui()
        
    def setup_ui(self):
        # Create main window
        self.root = tk.Tk()
        self.root.title("Three-Body Problem - Initial State Selector")
        self.root.geometry("1100x650")  # Kích thước cửa sổ phù hợp
        self.root.configure(bg=self.colors["light_bg"])
        self.root.resizable(True, True)
        
        # Configure ttk styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles
        self.style.configure("TFrame", background=self.colors["light_bg"])
        self.style.configure("TLabel", background=self.colors["light_bg"], 
                           foreground=self.colors["text"], font=self.fonts["body"])
        self.style.configure("TButton", background=self.colors["primary"],
                           foreground=self.colors["light_text"], font=self.fonts["button"], padding=6)
        self.style.map("TButton", background=[("active", self.colors["secondary"])])
        self.style.configure("TRadiobutton", background=self.colors["light_bg"],
                           foreground=self.colors["text"], font=self.fonts["body"])
        self.style.configure("TCombobox", background=self.colors["light_bg"],
                           fieldbackground=self.colors["light_bg"], foreground=self.colors["text"], 
                           font=self.fonts["body"])
        
        # Create main container with padding
        self.main_container = tk.Frame(self.root, bg=self.colors["light_bg"])
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        self.create_header()
        
        # Create main content area using grid layout
        self.content_frame = tk.Frame(self.main_container, bg=self.colors["light_bg"])
        self.content_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Configure grid layout
        self.content_frame.columnconfigure(0, weight=3)  # Left side (config)
        self.content_frame.columnconfigure(1, weight=1)  # Right side (buttons)
        self.content_frame.rowconfigure(0, weight=1)     # Main row
        
        # Create left panel for configuration
        self.left_panel = tk.Frame(self.content_frame, bg=self.colors["light_bg"])
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        
        # Create right panel for buttons and results
        self.right_panel = tk.Frame(self.content_frame, bg=self.colors["light_bg"])
        self.right_panel.grid(row=0, column=1, sticky="nsew")
        
        # Create configuration sections in left panel
        self.create_dimension_section()
        self.create_state_selector_section()
        self.create_entry_grid_section()
        
        # Create buttons and results in right panel
        self.create_action_buttons()
        self.create_results_section()
        
        # Initialize with default values
        self.on_dim_change()
        
    def create_header(self):
        """Create an attractive header section"""
        header_frame = tk.Frame(self.main_container, bg=self.colors["primary"], pady=10)
        header_frame.pack(fill=tk.X)
        
        title = tk.Label(
            header_frame, 
            text="Three-Body Problem Simulator", 
            font=("Helvetica", 18, "bold"),
            bg=self.colors["primary"],
            fg=self.colors["light_text"]
        )
        title.pack()
        
        subtitle = tk.Label(
            header_frame, 
            text="Initial State Configuration", 
            font=("Helvetica", 12),
            bg=self.colors["primary"],
            fg=self.colors["light_text"]
        )
        subtitle.pack()
        
    def create_dimension_section(self):
        """Create the dimension selection section"""
        dim_frame = tk.Frame(
            self.left_panel, 
            bg="white",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            padx=15,
            pady=10,
            relief=tk.FLAT
        )
        dim_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Section header
        header_label = tk.Label(
            dim_frame,
            text="Step 1: Choose Dimension",
            font=self.fonts["subheading"],
            bg=dim_frame["bg"],
            fg=self.colors["text"]
        )
        header_label.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Separator(dim_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Dimension selection
        self.dim_var = tk.StringVar(value="2D")
        
        rb_frame = tk.Frame(dim_frame, bg=dim_frame["bg"])
        rb_frame.pack(fill=tk.X)
        
        rb_2d = ttk.Radiobutton(
            rb_frame, 
            text="2D System (x, y, vx, vy)", 
            variable=self.dim_var, 
            value="2D", 
            command=self.on_dim_change
        )
        rb_2d.pack(side=tk.LEFT, padx=(0, 30))
        
        rb_3d = ttk.Radiobutton(
            rb_frame, 
            text="3D System (x, y, z, vx, vy, vz)", 
            variable=self.dim_var, 
            value="3D", 
            command=self.on_dim_change
        )
        rb_3d.pack(side=tk.LEFT)
        
    def create_state_selector_section(self):
        """Create the state selection section"""
        state_frame = tk.Frame(
            self.left_panel, 
            bg="white",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            padx=15,
            pady=10,
            relief=tk.FLAT
        )
        state_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Section header
        header_label = tk.Label(
            state_frame,
            text="Step 2: Select Predefined State",
            font=self.fonts["subheading"],
            bg=state_frame["bg"],
            fg=self.colors["text"]
        )
        header_label.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Separator(state_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # State selection
        combo_frame = tk.Frame(state_frame, bg=state_frame["bg"])
        combo_frame.pack(fill=tk.X, pady=5)
        
        label = tk.Label(
            combo_frame, 
            text="Choose a configuration:", 
            font=self.fonts["body"],
            bg=state_frame["bg"],
            fg=self.colors["text"]
        )
        label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.combo = ttk.Combobox(combo_frame, width=25, state="readonly")
        self.combo.pack(side=tk.LEFT, padx=(0, 10))
        self.combo.bind("<<ComboboxSelected>>", self.on_select)
        
        # Apply button
        apply_btn = self.create_modern_button(combo_frame, "Apply", self.succsess, self.colors["accent"])
        apply_btn.pack(side=tk.LEFT)
        
    def create_entry_grid_section(self):
        """Create the manual entry grid section"""
        entry_frame = tk.Frame(
            self.left_panel, 
            bg="white",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            padx=15,
            pady=10,
            relief=tk.FLAT
        )
        entry_frame.pack(fill=tk.BOTH, expand=True)
        
        # Section header
        header_label = tk.Label(
            entry_frame,
            text="Step 3: Manual Configuration",
            font=self.fonts["subheading"],
            bg=entry_frame["bg"],
            fg=self.colors["text"]
        )
        header_label.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Separator(entry_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Create container for entries
        self.entries_container = tk.Frame(entry_frame, bg=entry_frame["bg"])
        self.entries_container.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
    def update_entry_fields(self):
        """Update the entry fields based on dimension"""
        # Clear existing entries
        for widget in self.entries_container.winfo_children():
            widget.destroy()
            
        dim = self.selected_state["dim"]
        num_cols = 4 if dim == "2D" else 6
        labels = ["x", "y", "vx", "vy"] if dim == "2D" else ["x", "y", "z", "vx", "vy", "vz"]
        
        # Reset entries list
        self.entries = [[None for _ in range(num_cols)] for _ in range(3)]
        
        # Create grid layout for entries
        grid_frame = tk.Frame(self.entries_container, bg=self.entries_container["bg"])
        grid_frame.pack(fill=tk.BOTH, expand=True)
        
        # Configure grid columns
        grid_frame.grid_columnconfigure(0, weight=0, minsize=70)  # Body column
        for i in range(num_cols):
            grid_frame.grid_columnconfigure(i+1, weight=1, minsize=80)  # Parameter columns
            
        # Header row
        tk.Label(
            grid_frame, 
            text="Body", 
            font=self.fonts["subheading"],
            bg=grid_frame["bg"],
            fg=self.colors["text"]
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        for j, label in enumerate(labels):
            tk.Label(
                grid_frame, 
                text=label, 
                font=self.fonts["subheading"],
                bg=grid_frame["bg"],
                fg=self.colors["text"]
            ).grid(row=0, column=j+1, padx=5, pady=5)
        
        # Body colors for visual distinction
        body_colors = ["#f72585", "#4361ee", "#4cc9f0"]
        
        # Create entry rows for each body
        for i in range(3):
            # Body label with color
            body_label = tk.Label(
                grid_frame, 
                text=f"Body {i+1}", 
                font=self.fonts["body"],
                bg=body_colors[i],
                fg=self.colors["light_text"],
                padx=8,
                pady=6,
                relief=tk.FLAT
            )
            body_label.grid(row=i+1, column=0, padx=5, pady=3, sticky="ew")
            
            # Entry fields for each parameter
            for j in range(num_cols):
                entry = tk.Entry(
                    grid_frame, 
                    width=10,
                    font=self.fonts["body"],
                    bg="white",
                    relief=tk.SOLID,
                    bd=1,
                    justify=tk.CENTER
                )
                entry.grid(row=i+1, column=j+1, padx=3, pady=3, sticky="ew")
                entry.bind("<KeyRelease>", self.on_entry_change)
                
                # Add validation on focus out
                def validate_entry(event, entry=entry):
                    try:
                        value = entry.get()
                        if value.strip():
                            float(value)
                            entry.configure(bg="white")
                    except ValueError:
                        entry.configure(bg="#ffe6e6")  # Light red for invalid input
                
                entry.bind("<FocusOut>", validate_entry)
                
                # Store the entry widget
                self.entries[i][j] = entry
                
    def create_action_buttons(self):
        """Create action buttons section"""
        button_frame = tk.Frame(
            self.right_panel, 
            bg="white",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            padx=15,
            pady=15,
            relief=tk.FLAT
        )
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Section header
        header_label = tk.Label(
            button_frame,
            text="Actions",
            font=self.fonts["subheading"],
            bg=button_frame["bg"],
            fg=self.colors["text"]
        )
        header_label.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Separator(button_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Buttons stacked vertically
        confirm_btn = self.create_modern_button(
            button_frame, 
            "Confirm Configuration", 
            self.on_confirm, 
            self.colors["primary"],
            width=20
        )
        confirm_btn.pack(fill=tk.X, pady=(5, 10))
        
        reset_btn = self.create_modern_button(
            button_frame, 
            "Reset Values", 
            self.reset_entries, 
            self.colors["dark_bg"],
            width=20
        )
        reset_btn.pack(fill=tk.X)
        
        # Error message area
        self.error_frame = tk.Frame(button_frame, bg=button_frame["bg"])
        self.error_frame.pack(fill=tk.X, pady=(10, 0))
        
    def create_results_section(self):
        """Create the results section"""
        result_frame = tk.Frame(
            self.right_panel, 
            bg="white",
            highlightbackground=self.colors["border"],
            highlightthickness=1,
            padx=15,
            pady=15,
            relief=tk.FLAT
        )
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # Section header
        header_label = tk.Label(
            result_frame,
            text="Configuration Result",
            font=self.fonts["subheading"],
            bg=result_frame["bg"],
            fg=self.colors["text"]
        )
        header_label.pack(anchor=tk.W, pady=(0, 5))
        
        ttk.Separator(result_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        # Placeholder for when no result is available
        self.no_result_label = tk.Label(
            result_frame,
            text="No configuration confirmed yet.",
            font=self.fonts["body"],
            bg=result_frame["bg"],
            fg=self.colors["text"],
            justify=tk.CENTER
        )
        self.no_result_label.pack(expand=True, fill=tk.BOTH)
        
        # Container for actual results (initially hidden)
        self.result_container = tk.Frame(result_frame, bg=result_frame["bg"])
        
        # We'll populate this when results are available
        
    def update_results_view(self):
        """Update the results view with the current state"""
        # Clear existing content
        for widget in self.result_container.winfo_children():
            widget.destroy()
            
        if self.selected_state["state"] is None:
            self.no_result_label.pack(expand=True, fill=tk.BOTH)
            self.result_container.pack_forget()
            return
            
        # Hide the placeholder and show results
        self.no_result_label.pack_forget()
        self.result_container.pack(fill=tk.BOTH, expand=True)
        
        # Dimension badge
        dim_label = tk.Label(
            self.result_container,
            text=f"{self.selected_state['dim']} System",
            font=self.fonts["small"],
            bg=self.colors["accent"],
            fg=self.colors["light_text"],
            padx=8,
            pady=3,
            relief=tk.FLAT
        )
        dim_label.pack(anchor=tk.W, pady=(0, 10))
        
        # State vector display (compact)
        vector_label = tk.Label(
            self.result_container,
            text="State Vector:",
            font=self.fonts["body"],
            bg=self.result_container["bg"],
            fg=self.colors["text"],
            anchor=tk.W
        )
        vector_label.pack(fill=tk.X)
        
        # Format the state vector nicely but compact
        vector_str = "[" + ",\n ".join([f"{val:.4f}" for val in self.selected_state["state"]]) + "]"
        
        vector_text = tk.Text(
            self.result_container,
            height=6,
            width=20,
            font=("Courier", 9),
            bg="#f8f9fa",
            relief=tk.SOLID,
            bd=1,
            wrap=tk.WORD
        )
        vector_text.pack(fill=tk.X, pady=(0, 10))
        vector_text.insert(tk.END, vector_str)
        vector_text.configure(state="disabled")  # Make read-only
        
        # Body summary (very compact)
        dim = self.selected_state["dim"]
        num_cols = 4 if dim == "2D" else 6
        labels = ["x", "y", "vx", "vy"] if dim == "2D" else ["x", "y", "z", "vx", "vy", "vz"]
        
        # Body colors for visual distinction
        body_colors = ["#f72585", "#4361ee", "#4cc9f0"]
        
        # Create a compact summary for each body
        for i in range(3):
            body_frame = tk.Frame(
                self.result_container,
                bg=self.result_container["bg"],
                relief=tk.FLAT,
                highlightbackground=body_colors[i],
                highlightthickness=2,
                padx=5,
                pady=5
            )
            body_frame.pack(fill=tk.X, pady=(0, 5))
            
            # Body header
            tk.Label(
                body_frame,
                text=f"Body {i+1}",
                font=self.fonts["small"],
                bg=body_colors[i],
                fg=self.colors["light_text"],
                padx=5,
                pady=2
            ).pack(fill=tk.X, pady=(0, 5))
            
            # Parameter values in a compact grid
            values_frame = tk.Frame(body_frame, bg=body_frame["bg"])
            values_frame.pack(fill=tk.X)
            
            # Configure grid
            for j in range(2):  # 2 columns
                values_frame.columnconfigure(j*2, weight=0)  # Label column
                values_frame.columnconfigure(j*2+1, weight=1)  # Value column
            
            # Add parameters in a 2-column grid
            for j, label in enumerate(labels):
                row = j // 2
                col_start = (j % 2) * 2
                
                # Label
                tk.Label(
                    values_frame,
                    text=f"{label}:",
                    font=self.fonts["small"],
                    bg=values_frame["bg"],
                    fg=self.colors["text"],
                    anchor=tk.E
                ).grid(row=row, column=col_start, sticky="e", padx=(0, 3))
                
                # Value
                value = self.selected_state["state"][i * num_cols + j]
                tk.Label(
                    values_frame,
                    text=f"{value:.4f}",
                    font=("Courier", 9),
                    bg=values_frame["bg"],
                    fg=self.colors["text"],
                    anchor=tk.W
                ).grid(row=row, column=col_start+1, sticky="w", padx=(0, 10))
        
    def fill_entries(self, values, dim):
        """Fill the entry fields with values"""
        num_cols = 4 if dim == "2D" else 6
        for i in range(3):
            for j in range(num_cols):
                if i < len(self.entries) and j < len(self.entries[i]):
                    self.entries[i][j].delete(0, tk.END)
                    self.entries[i][j].insert(0, f"{values[i*num_cols + j]:.6f}")
                    
    def on_select(self, event=None):
        """Handle state selection"""
        name = self.combo.get()
        dim = self.selected_state["dim"]
        
        if name == "Random":
            state = generate_random_state(dim)
        elif name == "Random (Symmetric)":
            state = generate_random_state_sym(dim)
        elif name in named_states:
            state = named_states[name]
            expected_len = 3 * (4 if dim == "2D" else 6)
            if len(state) != expected_len:
                self.show_error(f"State '{name}' is not compatible with {dim}. Please choose another or switch dimension.")
                return
        else:
            return
            
        self.fill_entries(state, dim)
        self.clear_error()
        
    def on_entry_change(self, event=None):
        """Handle entry field changes"""
        self.combo.set("Custom")
        
    def on_confirm(self):
        """Validate and confirm the configuration"""
        try:
            state = []
            num_cols = 4 if self.selected_state["dim"] == "2D" else 6
            
            for i in range(3):
                for j in range(num_cols):
                    val_str = self.entries[i][j].get()
                    if val_str.strip() == "":
                        raise ValueError(f"Missing input for Body {i+1}, {['x','y','z','vx','vy','vz'][j]}")
                    val = float(val_str)
                    state.append(val)
                    
            self.selected_state["state"] = np.array(state)
            self.clear_error()
            
            # Show success message
            self.show_success("Configuration confirmed!")
            
            # Update results view
            self.update_results_view()
            
        except ValueError as e:
            self.show_error(f"Error: {str(e)}")
            
    def reset_entries(self):
        """Reset all entry fields"""
        for i in range(3):
            for j in range(len(self.entries[i]) if i < len(self.entries) else 0):
                if self.entries[i][j]:
                    self.entries[i][j].delete(0, tk.END)
                    self.entries[i][j].configure(bg="white")  # Reset background color
        self.combo.set("")
        self.clear_error()
        
    def on_dim_change(self):
        """Handle dimension change"""
        dim = self.dim_var.get()
        self.selected_state["dim"] = dim
        
        # Filter states based on dimension
        filtered_keys = [
            name for name, state in named_states.items()
            if (dim == "2D" and len(state) == 12) or (dim == "3D" and len(state) == 18)
        ]
        
        self.combo['values'] = filtered_keys + ["Random", "Random (Symmetric)", "Custom"]
        
        # Set default selection
        if dim == "2D" and "8-Figure" in filtered_keys:
            self.combo.set("8-Figure")
        elif dim == "3D" and "3D" in filtered_keys:
            self.combo.set("3D")
        else:
            self.combo.set("Random")
            
        self.update_entry_fields()
        self.on_select()
        
    def show_error(self, message):
        """Display an error message"""
        # Clear any existing error messages
        for widget in self.error_frame.winfo_children():
            widget.destroy()
            
        # Create error message with icon
        error_container = tk.Frame(self.error_frame, bg=self.colors["warning"], padx=10, pady=5)
        error_container.pack(fill=tk.X)
        
        error_label = tk.Label(
            error_container,
            text=message,
            font=self.fonts["small"],
            bg=error_container["bg"],
            fg=self.colors["light_text"],
            justify=tk.LEFT,
            wraplength=200
        )
        error_label.pack(fill=tk.X)
        
    def show_success(self, message):
        """Display a success message"""
        # Clear any existing messages
        for widget in self.error_frame.winfo_children():
            widget.destroy()
            
        # Create success message
        success_container = tk.Frame(self.error_frame, bg=self.colors["success"], padx=10, pady=5)
        success_container.pack(fill=tk.X)
        
        success_label = tk.Label(
            success_container,
            text=message,
            font=self.fonts["small"],
            bg=success_container["bg"],
            fg=self.colors["light_text"],
            justify=tk.LEFT
        )
        success_label.pack(fill=tk.X)
        
        # Auto-hide after 3 seconds
        self.root.after(3000, self.clear_error)
        
    def clear_error(self):
        """Clear any displayed error or success messages"""
        for widget in self.error_frame.winfo_children():
            widget.destroy()
        
    def run(self):
        """Run the application"""
        self.root.mainloop()
        return self.selected_state["state"], self.selected_state["dim"]
    
    def succsess(self):
        try:
            state = []
            num_cols = 4 if self.selected_state["dim"] == "2D" else 6
            for i in range(3):
                for j in range(num_cols):
                    val_str = self.entries[i][j].get()
                    if val_str.strip() == "":
                        raise ValueError("Missing input")
                    val = float(val_str)
                    state.append(val)
            self.selected_state["state"] = np.array(state)
            self.root.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers in all fields.")

def choose_state():
    """Main function to choose initial state"""
    selector = StateSelector()
    return selector.run()


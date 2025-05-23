import tkinter as tk
from tkinter import ttk, messagebox
from states import named_states, generate_random_state, generate_random_state_sym
import numpy as np

def choose_state():
    selected_state = {"state": None}

    def fill_entries(values):
        for i in range(3):
            for j in range(4):
                entries[i][j].delete(0, tk.END)
                entries[i][j].insert(0, f"{values[i*4 + j]:.6f}")

    def on_select(event=None):
        name = combo.get()
        if name == "Random":
            state = generate_random_state()
        elif name == "Random (Symmetric)":
            state = generate_random_state_sym()
        elif name in named_states:
            state = named_states[name]
        else:
            return
        fill_entries(state)

    def on_entry_change(event=None):
        combo.set("Custom")

    def on_confirm():
        try:
            state = []
            for i in range(3):
                for j in range(4):
                    val_str = entries[i][j].get()
                    if val_str.strip() == "":
                        raise ValueError("Missing input")
                    val = float(val_str)
                    state.append(val)
            selected_state["state"] = np.array(state)
            root.destroy()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers in all fields.")

    root = tk.Tk()
    root.title("Choose Initial State")

    # Combobox
    frame_top = tk.Frame(root)
    frame_top.pack(padx=10, pady=10)
    label = tk.Label(frame_top, text="Select a known initial state:")
    label.pack(side=tk.LEFT)
    combo = ttk.Combobox(frame_top, values=list(named_states.keys()) + ["Random", "Random (Symmetric)", "Custom"], width=25)
    combo.set("8-Figure")
    combo.pack(side=tk.LEFT, padx=5)
    combo.bind("<<ComboboxSelected>>", on_select)

    # Entry fields
    frame_entries = tk.Frame(root)
    frame_entries.pack(padx=10, pady=10)
    labels = ["x", "y", "vx", "vy"]
    entries = [[None for _ in range(4)] for _ in range(3)]

    for j, label in enumerate(["", *labels]):
        l = tk.Label(frame_entries, text=label, width=8)
        l.grid(row=0, column=j, padx=5, pady=2)

    for i in range(3):
        tk.Label(frame_entries, text=f"Body {i+1}", width=8).grid(row=i+1, column=0, padx=5, pady=5)
        for j in range(4):
            e = tk.Entry(frame_entries, width=10)
            e.grid(row=i+1, column=j+1, padx=5, pady=5)
            e.bind("<KeyRelease>", on_entry_change)
            entries[i][j] = e

    # Confirm button
    btn = tk.Button(root, text="OK", command=on_confirm)
    btn.pack(pady=10)

    # Preload initial state
    on_select()

    root.mainloop()
    return selected_state["state"]

import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np # For pd.notna
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.table import Table

class CsvToTableImageApp:
    def __init__(self, master):
        self.master = master
        master.title("EEG Results Table Image Generator")
        master.geometry("850x850") # Increased size for plot

        self.file_paths = {
            "baseline": tk.StringVar(),
            "hybrid_ae": tk.StringVar(),
            "mindreader": tk.StringVar()
        }

        # --- File Selection UI ---
        tk.Label(master, text="Select CSV Files:", font=('Arial', 14, 'bold')).pack(pady=(10,5))

        self.create_file_selector("Baseline CSP Results:", "baseline")
        self.create_file_selector("Hybrid AE Results (ConvAE):", "hybrid_ae")
        self.create_file_selector("MindReaderModel Results:", "mindreader")

        # --- Generate Button ---
        self.generate_button = tk.Button(master, text="Generate Table Image", command=self.generate_table_image, font=('Arial', 12, 'bold'), bg='lightblue')
        self.generate_button.pack(pady=15)

        # --- Frame for Matplotlib Canvas and Toolbar ---
        self.plot_frame = tk.Frame(master)
        self.plot_frame.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)
        self.canvas = None
        self.toolbar = None

        # --- Save Button ---
        self.save_button = tk.Button(master, text="Save Table Image...", command=self.save_image, font=('Arial', 12))
        self.save_button.pack(pady=10)
        self.save_button.config(state=tk.DISABLED)

        self.fig = None # To store the matplotlib figure

    def create_file_selector(self, label_text, file_key):
        frame = tk.Frame(self.master)
        frame.pack(fill=tk.X, padx=20, pady=5)

        label = tk.Label(frame, text=label_text, width=25, anchor='w', font=('Arial', 11))
        label.pack(side=tk.LEFT)

        entry = tk.Entry(frame, textvariable=self.file_paths[file_key], width=50, font=('Arial', 10))
        entry.pack(side=tk.LEFT, padx=5)

        button = tk.Button(frame, text="Browse...", command=lambda fk=file_key: self.browse_file(fk), font=('Arial', 10))
        button.pack(side=tk.LEFT)

    def browse_file(self, file_key):
        filename = filedialog.askopenfilename(
            initialdir=".",
            title=f"Select {file_key.replace('_', ' ').title()} CSV File",
            filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
        )
        if filename:
            self.file_paths[file_key].set(filename)

    def clear_plot_frame(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
        self.fig = None # Clear figure reference

    def generate_table_image(self):
        self.clear_plot_frame() # Clear previous plot
        self.save_button.config(state=tk.DISABLED)

        path_baseline = self.file_paths["baseline"].get()
        path_hybrid_ae = self.file_paths["hybrid_ae"].get()
        path_mindreader = self.file_paths["mindreader"].get()

        if not all([path_baseline, path_hybrid_ae, path_mindreader]):
            messagebox.showerror("Error", "Please select all three CSV files.")
            return

        try:
            baseline_df_raw = pd.read_csv(path_baseline)
            hybrid_ae_df_raw = pd.read_csv(path_hybrid_ae)
            mindreader_df_raw = pd.read_csv(path_mindreader)

            # --- Processing logic ---
            # Baseline CSP
            baseline_df_raw['subject_num'] = baseline_df_raw['subject'].str.extract(r'(\d+)').astype(int)
            baseline_avg_csp_df = baseline_df_raw.groupby('subject_num')['mean_accuracy'].mean().reset_index()
            baseline_avg_csp_df.rename(columns={'mean_accuracy': 'Baseline CSP (Acc %)'}, inplace=True) # Shorter name
            baseline_avg_csp_df['Baseline CSP (Acc %)'] *= 100

            # Hybrid AE
            hybrid_ae_df_raw['subject_num'] = hybrid_ae_df_raw['subject'].str.extract(r'(\d+)').astype(int)
            hybrid_avg_ae_df = hybrid_ae_df_raw.groupby('subject_num')['mean_accuracy'].mean().reset_index()
            hybrid_avg_ae_df.rename(columns={'mean_accuracy': 'Hybrid AE (Acc %)'}, inplace=True) # Shorter name
            hybrid_avg_ae_df['Hybrid AE (Acc %)'] *= 100

            # MindReaderModel
            mindreader_df_raw['subject_num'] = mindreader_df_raw['subject'].str.extract(r'(\d+)').astype(int)
            mindreader_model_df = mindreader_df_raw[mindreader_df_raw['pipeline'] == 'MindReaderModel'].copy()
            mindreader_model_df.rename(columns={'mean_accuracy': 'MindReader (Acc %)'}, inplace=True) # Shorter name
            mindreader_model_df['MindReader (Acc %)'] *= 100
            mindreader_model_df = mindreader_model_df[['subject_num', 'MindReader (Acc %)']]

            # Merge
            all_subjects = pd.DataFrame({'subject_num': range(1, 10)})
            merged_df = pd.merge(all_subjects, baseline_avg_csp_df, on='subject_num', how='left')
            merged_df = pd.merge(merged_df, hybrid_avg_ae_df, on='subject_num', how='left')
            merged_df = pd.merge(merged_df, mindreader_model_df, on='subject_num', how='left')
            merged_df.sort_values('subject_num', inplace=True)
            merged_df.rename(columns={'subject_num': 'Subject ID'}, inplace=True)

            # Prepare data for table (including average row)
            display_df = merged_df.copy()
            avg_baseline_csp = display_df['Baseline CSP (Acc %)'].mean()
            avg_hybrid_ae = display_df['Hybrid AE (Acc %)'].mean()
            avg_mindreader = display_df['MindReader (Acc %)'].mean()

            # Format numbers to 1 decimal place for display in table
            for col in ['Baseline CSP (Acc %)', 'Hybrid AE (Acc %)', 'MindReader (Acc %)']:
                display_df[col] = display_df[col].apply(lambda x: f"{x:.1f}" if pd.notna(x) else "N/A")

            avg_row = pd.DataFrame([{
                'Subject ID': '**Average**',
                'Baseline CSP (Acc %)': f"**{avg_baseline_csp:.1f}**" if pd.notna(avg_baseline_csp) else "N/A",
                'Hybrid AE (Acc %)': f"**{avg_hybrid_ae:.1f}**" if pd.notna(avg_hybrid_ae) else "N/A",
                'MindReader (Acc %)': f"**{avg_mindreader:.1f}**" if pd.notna(avg_mindreader) else "N/A"
            }])
            display_df = pd.concat([display_df, avg_row], ignore_index=True)
            display_df['Subject ID'] = display_df['Subject ID'].astype(str) # Ensure Subject ID is string for table

            # --- Create Matplotlib Table Plot ---
            self.fig, ax = plt.subplots(figsize=(8, 4)) # Adjust figsize as needed
            ax.axis('tight')
            ax.axis('off')
            
            table_data = [display_df.columns.tolist()] + display_df.values.tolist()
            
            mpl_table = ax.table(cellText=table_data,
                                 colLabels=None, # Already included in table_data
                                 loc='center',
                                 cellLoc='center')

            mpl_table.auto_set_font_size(False)
            mpl_table.set_fontsize(10)
            mpl_table.scale(1.2, 1.2) # Adjust scale for better appearance

            # Style header and average row
            for i, key in enumerate(table_data[0]):
                mpl_table[(0, i)].set_text_props(weight='bold', color='white')
                mpl_table[(0, i)].set_facecolor('royalblue')
                if table_data[-1][i].startswith('**'): # Check for average row
                    mpl_table[(len(table_data)-1, i)].set_text_props(weight='bold')
                    mpl_table[(len(table_data)-1, i)].get_text().set_text(table_data[-1][i].replace('**','')) # Remove markdown bold

            # Embed in Tkinter
            if self.canvas: self.canvas.get_tk_widget().destroy()
            if self.toolbar: self.toolbar.destroy()

            self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
            self.toolbar.update()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            self.save_button.config(state=tk.NORMAL) # Enable save button

        except FileNotFoundError as e:
            messagebox.showerror("File Not Found", f"Error: Could not find one of the CSV files.\n{e}")
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred while processing the files:\n{e}")


    def save_image(self):
        if self.fig is None:
            messagebox.showerror("Error", "No table image to save. Please generate first.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
            title="Save Table Image As..."
        )
        if filepath:
            try:
                self.fig.savefig(filepath, bbox_inches='tight', dpi=300)
                messagebox.showinfo("Success", f"Table image saved to:\n{filepath}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Could not save image:\n{e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = CsvToTableImageApp(root)
    root.mainloop()

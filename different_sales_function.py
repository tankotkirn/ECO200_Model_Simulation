import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def linear_sales_function(e, c):
    return sigmoid( e) * c

def quadratic_sales_function(e, c):
    return c * sigmoid(e**2)

def logarithmic_sales_function(e, c):
    return c * sigmoid(np.log(e + 1))  # Adding 1 to avoid log(0)

def exponential_sales_function(e, c):
    return c * sigmoid(np.exp(0.1 * e))

def agent_objective(params, sales_function):
    c, e = params
    s = sales_function(e, c)
    agent_profit = -(c * s - 1 * e)  # Negative agent's profit for minimization
    total_sales = s * (1 - sales_function(e, c))  # Total sales
    agent_profit_percentage = (agent_profit / total_sales) * 100 if total_sales != 0 else 0
    return agent_profit_percentage, agent_profit

def company_profit(e, c, sales_function):
    return round(sales_function(e, c) * (1 - sales_function(e, c)) * 100, 2)  # Round to 2 digits and add %

def plot_profit_surface(ax, commission_range, effort_range, values, title, xlabel, ylabel, zlabel, cmap):
    commission_values, effort_values = np.meshgrid(commission_range, effort_range)

    ax.plot_surface(commission_values, effort_values, values, cmap=cmap, edgecolor='k')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

def create_gui():
    root = tk.Tk()
    root.title("Incentive Optimization: Aligning Commission and Effort in Principal-Agent Dynamics")

    # Create variables
    commission_var = tk.DoubleVar(value=0.5)
    effort_var = tk.DoubleVar(value=5.0)
    agent_profit_var = tk.DoubleVar(value=0.0)
    company_profit_var = tk.StringVar(value="0.00%")  # Initialized with 0.00%
    selected_function_var = tk.StringVar(value="Linear")

    # Create GUI components
    frame = ttk.Frame(root, padding="10")
    frame.grid(column=0, row=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    frame.columnconfigure(0, weight=1)
    frame.rowconfigure(0, weight=1)

    # Create Matplotlib Figure and Axes with adjusted layout
    fig = plt.Figure(figsize=(12, 4), tight_layout={'w_pad': 3.0})
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')  # New subplot for correlation

    # Create Tkinter Canvas
    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    # Callback functions
    def show_error(message):
        messagebox.showerror("Error", message)

    def validate_input():
        commission = commission_var.get()
        effort = effort_var.get()

        if not (0.1 <= commission <= 0.9):
            show_error("Commission value must be between 0.1 and 0.9.")
            return False

        if not (0 <= effort <= 10):
            show_error("Effort value must be between 0 and 10.")
            return False

        return True

    def update_plots():
        if not validate_input():
            return

        commission = commission_var.get()
        effort = effort_var.get()

        selected_function = selected_function_var.get()

        if selected_function == "Linear":
            sales_function = linear_sales_function
        elif selected_function == "Quadratic":
            sales_function = quadratic_sales_function
        elif selected_function == "Logarithmic":
            sales_function = logarithmic_sales_function
        elif selected_function == "Exponential":
            sales_function = exponential_sales_function
        else:
            show_error("Invalid sales function selected.")
            return

        agent_profit_percentage, agent_profit = agent_objective([commission, effort], sales_function)
        agent_profit_var.set(f"{agent_profit_percentage:.2f}")  # Corrected line to display agent profit without % symbol

        company_profit_percentage = company_profit(effort, commission, sales_function)
        company_profit_var.set(f"{company_profit_percentage:.2f}%")  # Format with 2 decimal places and %

        commission_range = np.linspace(0.1, 0.9, 100)
        effort_range = np.linspace(0, 10, 100)

        agent_profit_values = np.zeros((len(effort_range), len(commission_range)))
        company_profit_values = np.zeros((len(effort_range), len(commission_range)))

        for i in range(len(commission_range)):
            for j in range(len(effort_range)):
                agent_profit_values[j, i], _ = agent_objective([commission_range[i], effort_range[j]], sales_function)
                company_profit_values[j, i] = company_profit(effort_range[j], commission_range[i], sales_function)

        ax1.clear()
        plot_profit_surface(ax1, commission_range, effort_range, agent_profit_values, 'Agent Profit',
                            'Commission', 'Effort', 'Agent Profit', 'viridis')

        ax2.clear()
        plot_profit_surface(ax2, commission_range, effort_range, company_profit_values, 'Company Profit Percentage',
                            'Commission', 'Effort', 'Company Profit Percentage', 'plasma')

        # Scatter plot for correlation
        ax3.clear()
        ax3.scatter([commission], [effort], [agent_profit_percentage], c='blue', marker='o', label='Agent Profit')
        ax3.scatter([commission], [effort], [company_profit_percentage], c='red', marker='o', label='Company Profit %')
        ax3.set_xlabel('Commission')
        ax3.set_ylabel('Effort')
        ax3.set_zlabel('Profit (Percentage)')
        ax3.legend()

        canvas.draw()

    # GUI components
    ttk.Label(frame, text="Commission:").grid(row=1, column=0, sticky=tk.W, padx=5)
    ttk.Entry(frame, textvariable=commission_var).grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
    ttk.Label(frame, text="Effort:").grid(row=2, column=0, sticky=tk.W, padx=5)
    ttk.Entry(frame, textvariable=effort_var).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=5)

    # Dropdown menu for selecting sales function
    ttk.Label(frame, text="Select Sales Function:").grid(row=3, column=0, sticky=tk.W, padx=5)
    sales_function_dropdown = ttk.Combobox(frame, textvariable=selected_function_var, values=["Linear", "Quadratic", "Logarithmic", "Exponential"])
    sales_function_dropdown.grid(row=3, column=1, sticky=(tk.W, tk.E), padx=5)
    sales_function_dropdown.set("Linear")

    ttk.Button(frame, text="Calculate & Update Plots", command=update_plots).grid(row=4, column=0, columnspan=2, pady=10)

    # Labels for displaying values
    ttk.Label(frame, text="Agent Profit:").grid(row=5, column=0, sticky=tk.W, padx=5)
    ttk.Label(frame, textvariable=agent_profit_var).grid(row=5, column=1, sticky=tk.W, padx=5)

    ttk.Label(frame, text="Company Profit:").grid(row=6, column=0, sticky=tk.W, padx=5)
    ttk.Label(frame, textvariable=company_profit_var).grid(row=6, column=1, sticky=tk.W, padx=5)

    # Initial plots
    commission_range = np.linspace(0.1, 0.9, 100)
    effort_range = np.linspace(0, 10, 100)
    update_plots()

    root.mainloop()

if __name__ == "__main__":
    create_gui()

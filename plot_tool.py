import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil

PLOTS_DIR = "plots"

def clear_plots_directory():
    if os.path.exists(PLOTS_DIR):
        shutil.rmtree(PLOTS_DIR)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print(f"Cleared and recreated directory: {PLOTS_DIR}")

def generate_plot(df, x_col: str, y_col: str = None, plot_type: str = "scatter", title: str = "Plot", filename: str = "plot.png"):
    plt.figure(figsize=(10, 6))
    
    if plot_type == "scatter":
        if y_col is None:
            raise ValueError("y_col is required for scatter plot.")
        sns.scatterplot(data=df, x=x_col, y=y_col)
    elif plot_type == "line":
        if y_col is None:
            raise ValueError("y_col is required for line plot.")
        sns.lineplot(data=df, x=x_col, y=y_col)
    elif plot_type == "bar":
        if y_col is None:
            raise ValueError("y_col is required for bar plot.")
        sns.barplot(data=df, x=x_col, y=y_col)
    elif plot_type == "hist":
        sns.histplot(data=df, x=x_col, kde=True)
    elif plot_type == "box":
        sns.boxplot(data=df, x=x_col, y=y_col)
    elif plot_type == "kde":
        sns.kdeplot(data=df, x=x_col, y=y_col, fill=True)
    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    plt.title(title)
    plt.xlabel(x_col)
    if y_col:
        plt.ylabel(y_col)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    return f"Plot '{filename}' generated successfully and saved to '{PLOTS_DIR}'. Consider creating plots whenever extra visualization can be added."

def generate_aggregated_plot(x_data: list, y_data: list, x_label: str = "X-axis", y_label: str = "Y-axis", plot_type: str = "bar", title: str = "Aggregated Plot", filename: str = "aggregated_plot.png"):
    plt.figure(figsize=(10, 6))
    
    if plot_type == "bar":
        sns.barplot(x=x_data, y=y_data)
    elif plot_type == "line":
        sns.lineplot(x=x_data, y=y_data)
    elif plot_type == "scatter":
        sns.scatterplot(x=x_data, y=y_data)
    else:
        raise ValueError(f"Unsupported plot type for aggregated data: {plot_type}. Supported types are 'bar', 'line', 'scatter'.")

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")
    return f"Aggregated plot '{filename}' generated successfully and saved to '{PLOTS_DIR}'. Consider creating plots whenever extra visualization can be added."

def generate_subplots(df, plot_specs: list, filename: str = "subplots.png"):
    num_plots = len(plot_specs)
    if num_plots == 0:
        return "No plot specifications provided for subplots."

    # Determine grid size (simple square-ish layout)
    rows = int(num_plots**0.5)
    cols = (num_plots + rows - 1) // rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, spec in enumerate(plot_specs):
        if i >= len(axes):
            break
        ax = axes[i]
        x_col = spec.get('x_col')
        y_col = spec.get('y_col')
        plot_type = spec.get('plot_type', 'scatter')
        title = spec.get('title', 'Subplot')

        if not x_col:
            print(f"Warning: x_col missing for subplot {i+1}. Skipping.")
            continue

        try:
            if plot_type == "scatter":
                if y_col is None:
                    raise ValueError("y_col is required for scatter plot.")
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            elif plot_type == "line":
                if y_col is None:
                    raise ValueError("y_col is required for line plot.")
                sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
            elif plot_type == "bar":
                if y_col is None:
                    raise ValueError("y_col is required for bar plot.")
                sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
            elif plot_type == "hist":
                sns.histplot(data=df, x=x_col, kde=True, ax=ax)
            elif plot_type == "box":
                sns.boxplot(data=df, x=x_col, y=y_col, ax=ax)
            elif plot_type == "kde":
                sns.kdeplot(data=df, x=x_col, y=y_col, fill=True, ax=ax)
            else:
                print(f"Warning: Unsupported plot type '{plot_type}' for subplot {i+1}. Skipping.")
                continue

            ax.set_title(title)
            ax.set_xlabel(x_col)
            if y_col:
                ax.set_ylabel(y_col)
        except Exception as e:
            print(f"Error generating subplot {i+1}: {e}")
            ax.set_title(f"Error: {title}")
            ax.text(0.5, 0.5, f"Plot Error: {e}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red')

    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, filename)
    plt.savefig(plot_path)
    plt.close(fig)
    print(f"Subplots saved to {plot_path}")
    return f"Subplots '{filename}' generated successfully and saved to '{PLOTS_DIR}'. Consider creating plots whenever extra visualization can be added."

if __name__ == "__main__":
    clear_plots_directory()
    data = {'A': [1, 2, 3, 4, 5], 'B': [5, 4, 3, 2, 1], 'C': ['X', 'Y', 'X', 'Y', 'X']}
    test_df = pd.DataFrame(data)

    generate_plot(test_df, 'A', 'B', plot_type='scatter', title='Scatter Plot of A vs B', filename='scatter_A_B.png')

    generate_plot(test_df, 'A', plot_type='hist', title='Histogram of A', filename='hist_A.png')

    x_agg = ['Category A', 'Category B', 'Category C']
    y_agg = [10, 25, 15]
    generate_aggregated_plot(x_agg, y_agg, x_label='Category', y_label='Value', plot_type='bar', title='Aggregated Bar Plot', filename='aggregated_bar.png')

    subplot_specs = [
        {'x_col': 'A', 'y_col': 'B', 'plot_type': 'scatter', 'title': 'Subplot 1: A vs B'},
        {'x_col': 'A', 'plot_type': 'hist', 'title': 'Subplot 2: Histogram of A'},
        {'x_col': 'C', 'y_col': 'A', 'plot_type': 'bar', 'title': 'Subplot 3: C vs A'}
    ]
    generate_subplots(test_df, subplot_specs, filename='multiple_plots.png')

import json
import os

NOTEBOOK_PATH = "notebooks/train_colab.ipynb"

def update_notebook():
    if not os.path.exists(NOTEBOOK_PATH):
        print(f"Error: {NOTEBOOK_PATH} not found.")
        return

    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Define new cells
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Backtesting"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from src.backtest import Backtester\n",
                "\n",
                "# Initialize and Run Backtest\n",
                "backtester = Backtester()\n",
                "backtester.load_models()\n",
                "backtester.run_backtest(\"EURUSD\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from IPython.display import Image\n",
                "Image(filename='data/EURUSD_backtest.png')"
            ]
        }
    ]

    # Check if backtesting section already exists to avoid duplication
    existing_sources = [
        "".join(cell["source"]).strip() 
        for cell in nb["cells"] 
        if cell["cell_type"] == "markdown"
    ]
    
    if "## 5. Backtesting" in existing_sources:
        print("Backtesting section already exists. Skipping update.")
        return

    # Append new cells
    nb["cells"].extend(new_cells)

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print(f"Successfully added backtesting cells to {NOTEBOOK_PATH}")

if __name__ == "__main__":
    update_notebook()

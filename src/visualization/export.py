from pathlib import Path

def get_output_dir(folder: str = "../Figures") -> Path:
   # Convert the folder argument into a Path object 
    out_dir = Path(folder)

    # Create the directory if it doesn't already exist
    # parents=True ensures nested folders are created
    # exist_ok=True prevents errors if the folder already exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Return the folder location as a Path object
    return out_dir

def save_plot(fig, name: str, folder: str = "../Figures") -> None:
    # Ensure the output directory exists (uses the function above)
    out_dir = get_output_dir(folder)

    # Construct full file path: e.g., Figures/plotname.png
    path = out_dir / f"{name}.png"

     # Save the figure with high resolution and tight layout
    fig.savefig(path, dpi=300, bbox_inches="tight")

    # Print where the file was saved (useful for logging/debugging)
    print(f"Saved figure: {path}")
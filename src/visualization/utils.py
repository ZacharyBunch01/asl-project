from pathlib import Path

def get_output_dir(folder: str = "../Figures") -> Path:
    """Return the output directory Path, creating it if needed"""
    out_dir = Path(folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_plot(fig, name: str, folder: str = "../Figures") -> None:
    """Save a matplotlib Figure to the given folder the given base name"""
    out_dir = get_output_dir(folder)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved figure: {path}")
"""Loss and validation-accuracy panels inspired by paper Figures 4 and 5."""

from pathlib import Path


def plot_history(history, output, *, model_name, synthetic):
    """Render measured history only; a one-step smoke is not a training curve."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not history:
        raise ValueError("Cannot plot empty training history")
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].plot(epochs, [item["train_loss"] for item in history], "o-", label="Train")
    axes[0].plot(epochs, [item["val_loss"] for item in history], "s-", label="Validation")
    axes[0].set(xlabel="Completed epoch", ylabel="Weighted cross-entropy", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, [item["val_pixel_accuracy"] for item in history], "o-", color="#167f72")
    axes[1].set(xlabel="Completed epoch", ylabel="Pixel accuracy", ylim=(0, 1), title="Validation accuracy")
    for axis in axes:
        axis.grid(alpha=.2)
        axis.set_xticks(epochs if len(epochs) < 12 else epochs[::max(1, len(epochs)//10)])
    qualifier = "Synthetic CPU smoke only — not paper results" if synthetic else "Measured local training history"
    fig.suptitle(f"{model_name}: {qualifier}", fontsize=11)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output

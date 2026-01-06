import matplotlib.pyplot as plt


class YieldCurveVisualizer:
    """Plots and compares yield curves."""

    def plot_curves(self, curves, title="Indian G-Sec Yield Curve Comparison"):
        plt.figure(figsize=(10, 6))

        for label, curve in curves.items():
            maturities = [int(x.replace('Y', '')) for x in curve.index]
            plt.plot(maturities, curve.values, marker='o', label=label)

        plt.xlabel("Maturity (Years)")
        plt.ylabel("Yield (%)")
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

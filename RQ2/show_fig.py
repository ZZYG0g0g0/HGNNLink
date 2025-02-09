import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

def plot_results():
    methods = ["word2vec", "BERT", "Roberta", "XLNET", "ALBERT", "HGNNLink"]

    all_data = []
    for method in methods:
        df = pd.read_excel(f"./{method}.xlsx")
        df["Method"] = method
        all_data.append(df)
    all_data = pd.concat(all_data, ignore_index=True)

    sns.set(style="whitegrid", font_scale=1.2)

    fig, axes = plt.subplots(3, 1, figsize=(18, 15))  # 宽一些，留给图例
    metrics = ["Precision", "Recall", "F1"]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(
            x="Dataset",
            y=metric,
            hue="Method",
            data=all_data,
            ax=ax,
            palette="Set2"
        )
        ax.set_title(f"{metric} Comparison by Dataset", fontsize=14)
        ax.set_xlabel("Dataset")
        ax.set_ylabel(metric)

        # 将图例放到绘图区右侧外面
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0
        )

    # 适当减少绘图区在横向的比例，从而给图例留下空间
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig("results_change.pdf", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_results()

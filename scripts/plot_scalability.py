
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Project paths
ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "results" / "scalability.csv"
OUTPUT_DIR = ROOT / "results" / "graphs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
df["corpus_size"] = pd.to_numeric(df["corpus_size"])

# Graph 1: Mean search latency
plt.figure(figsize=(9, 5))
for index_type, group in df.groupby("index_type"):
    group = group.sort_values("corpus_size")
    plt.plot(
        group["corpus_size"],
        group["mean_latency_ms"],
        marker="o",
        label=index_type,
    )
plt.xlabel("Number of Documents")
plt.ylabel("Mean Search Latency (ms)")
plt.title("Search Latency vs. Dataset Size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scalability_latency.png", dpi=300)
plt.close()

# Graph 2: Estimated throughput
plt.figure(figsize=(9, 5))
for index_type, group in df.groupby("index_type"):
    group = group.sort_values("corpus_size")
    plt.plot(
        group["corpus_size"],
        group["throughput_qps"],
        marker="o",
        label=index_type,
    )
plt.xlabel("Number of Documents")
plt.ylabel("Estimated Throughput (queries/second)")
plt.title("Estimated Throughput vs. Dataset Size")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scalability_throughput.png", dpi=300)
plt.close()

# Graph 3: Recall@10
plt.figure(figsize=(9, 5))
for index_type, group in df.groupby("index_type"):
    group = group.sort_values("corpus_size")
    plt.plot(
        group["corpus_size"],
        group["mean_recall_at_k"] * 100,
        marker="o",
        label=index_type,
    )
plt.xlabel("Number of Documents")
plt.ylabel("Recall@10 (%)")
plt.title("Retrieval Recall vs. Dataset Size")
plt.ylim(0, 105)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "scalability_recall.png", dpi=300)
plt.close()

print(f"Graphs saved to: {OUTPUT_DIR}")
for path in sorted(OUTPUT_DIR.glob("scalability_*.png")):
    print(f" - {path.name}")

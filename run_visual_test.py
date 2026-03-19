import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, ".")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

data = np.load("data/mnist_cache.npz", allow_pickle=True)
all_images = data["data"]
all_labels = data["target"]

rng = np.random.default_rng(42)


def get_samples(digit, n=5):
    mask = all_labels == str(digit)
    imgs = all_images[mask]
    chosen = rng.choice(len(imgs), size=n, replace=False)
    results = []
    for i in chosen:
        img28 = imgs[i].reshape(28, 28)
        img8 = np.array(Image.fromarray(img28.astype(np.uint8)).resize((8, 8), Image.BILINEAR))
        results.append((img28, img8))
    return results


fig, axes = plt.subplots(10, 12, figsize=(20, 18))
fig.suptitle("What Does the Fly Brain SEE? — MNIST at 28x28 vs 8x8 Resolution",
             fontsize=16, fontweight="bold")

for digit in range(10):
    samples = get_samples(digit, n=3)
    for s_idx, (img28, img8) in enumerate(samples):
        col_28 = s_idx * 2
        col_8 = s_idx * 2 + 1

        axes[digit, col_28 * 2].imshow(img28, cmap="gray", interpolation="nearest")
        axes[digit, col_28 * 2].set_xticks([])
        axes[digit, col_28 * 2].set_yticks([])
        if s_idx == 0:
            axes[digit, col_28 * 2].set_ylabel(f"Digit {digit}", fontsize=12, fontweight="bold")

        axes[digit, col_28 * 2 + 1].imshow(img8, cmap="gray", interpolation="nearest")
        axes[digit, col_28 * 2 + 1].set_xticks([])
        axes[digit, col_28 * 2 + 1].set_yticks([])

    for s_idx, (img28, img8) in enumerate(samples):
        rates = img8.flatten() / 255.0 * 100.0
        col = 6 + s_idx * 2

        axes[digit, col].imshow(rates.reshape(8, 8), cmap="hot", vmin=0, vmax=100,
                               interpolation="nearest")
        axes[digit, col].set_xticks([])
        axes[digit, col].set_yticks([])

        spikes = np.zeros(64)
        for pn in range(64):
            spikes[pn] = np.sum(rng.random(2000) < (rates[pn] * 0.00005))
        axes[digit, col + 1].imshow(spikes.reshape(8, 8), cmap="YlOrRd", interpolation="nearest")
        axes[digit, col + 1].set_xticks([])
        axes[digit, col + 1].set_yticks([])

axes[0, 0].set_title("28x28", fontsize=9)
axes[0, 1].set_title("8x8", fontsize=9)
axes[0, 2].set_title("28x28", fontsize=9)
axes[0, 3].set_title("8x8", fontsize=9)
axes[0, 4].set_title("28x28", fontsize=9)
axes[0, 5].set_title("8x8", fontsize=9)
axes[0, 6].set_title("Rate Hz", fontsize=9)
axes[0, 7].set_title("Spikes", fontsize=9)
axes[0, 8].set_title("Rate Hz", fontsize=9)
axes[0, 9].set_title("Spikes", fontsize=9)
axes[0, 10].set_title("Rate Hz", fontsize=9)
axes[0, 11].set_title("Spikes", fontsize=9)

plt.tight_layout()
fig.savefig("docs/figures/visual_test_overview.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Overview saved")


fig2, axes2 = plt.subplots(2, 5, figsize=(18, 7))
fig2.suptitle("Can Fly Eyes Tell Digits Apart? — Average 8x8 Activation Per Digit",
              fontsize=14, fontweight="bold")

digit_means = []
for digit in range(10):
    mask = all_labels == str(digit)
    imgs = all_images[mask][:200]
    resized = []
    for img in imgs:
        img8 = np.array(Image.fromarray(img.reshape(28, 28).astype(np.uint8)).resize((8, 8), Image.BILINEAR))
        resized.append(img8.flatten() / 255.0)
    mean_img = np.mean(resized, axis=0).reshape(8, 8)
    digit_means.append(mean_img)

    r, c = digit // 5, digit % 5
    im = axes2[r, c].imshow(mean_img, cmap="hot", vmin=0, vmax=0.8, interpolation="nearest")
    axes2[r, c].set_title(f"Digit {digit}", fontsize=12, fontweight="bold")
    axes2[r, c].set_xticks([])
    axes2[r, c].set_yticks([])

plt.tight_layout()
fig2.savefig("docs/figures/visual_test_means.png", dpi=200, bbox_inches="tight")
plt.close(fig2)
print("Means saved")


fig3, ax3 = plt.subplots(figsize=(10, 8))
fig3.suptitle("Digit Similarity Matrix — How Similar Do Digits Look at 8x8?",
              fontsize=14, fontweight="bold")

similarity = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        a = digit_means[i].flatten()
        b = digit_means[j].flatten()
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        similarity[i, j] = cos_sim

im = ax3.imshow(similarity, cmap="RdYlGn_r", vmin=0.5, vmax=1.0, interpolation="nearest")
ax3.set_xticks(range(10))
ax3.set_yticks(range(10))
ax3.set_xlabel("Digit", fontsize=12)
ax3.set_ylabel("Digit", fontsize=12)
plt.colorbar(im, ax=ax3, label="Cosine Similarity")

for i in range(10):
    for j in range(10):
        color = "white" if similarity[i, j] > 0.85 else "black"
        ax3.text(j, i, f"{similarity[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color=color, fontweight="bold")

plt.tight_layout()
fig3.savefig("docs/figures/visual_test_similarity.png", dpi=200, bbox_inches="tight")
plt.close(fig3)
print("Similarity saved")

print("\nMost confusable pairs (similarity > 0.90):")
pairs = []
for i in range(10):
    for j in range(i+1, 10):
        if similarity[i, j] > 0.90:
            pairs.append((i, j, similarity[i, j]))
pairs.sort(key=lambda x: -x[2])
for i, j, s in pairs:
    print(f"  {i} vs {j}: {s:.3f}")

print("\nMost distinguishable pairs (similarity < 0.75):")
pairs2 = []
for i in range(10):
    for j in range(i+1, 10):
        if similarity[i, j] < 0.75:
            pairs2.append((i, j, similarity[i, j]))
pairs2.sort(key=lambda x: x[2])
for i, j, s in pairs2:
    print(f"  {i} vs {j}: {s:.3f}")

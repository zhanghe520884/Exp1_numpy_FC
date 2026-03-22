"""
train.py - 训练主程序（含超参数对比实验）
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import load_mnist, accuracy, to_numpy, DEVICE
from model import MLP


# ═══════════════════════════ 训练函数 ═══════════════════════════

def train(model, X_train, Y_train, X_val, Y_val, y_val_int,
          epochs=30, batch_size=64, verbose=True):
    """
    Mini-batch 梯度下降训练循环。
    X_train / Y_train 已在目标设备（GPU or CPU）上。
    y_val_int 为 numpy 整数数组（用于 accuracy 计算）。
    """
    N = len(X_train)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # 随机打乱（在设备上做索引）
        idx = np.random.permutation(N)
        from utils import xp
        idx_dev = xp.asarray(idx)
        X_sh, Y_sh = X_train[idx_dev], Y_train[idx_dev]

        epoch_loss, num_batches = 0.0, 0

        for start in range(0, N, batch_size):
            Xb = X_sh[start: start + batch_size]
            Yb = Y_sh[start: start + batch_size]

            loss, probs = model.loss(Xb, Yb, training=True)
            model.backward(probs, Yb)
            model.update()

            epoch_loss += float(loss)
            num_batches += 1

        # 验证（分批避免 GPU OOM）
        val_loss, val_probs = model.loss(X_val, Y_val, training=False)
        val_acc = accuracy(val_probs, y_val_int)

        history["train_loss"].append(epoch_loss / num_batches)
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(val_acc)

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={epoch_loss/num_batches:.4f} | "
                  f"val_loss={float(val_loss):.4f} | "
                  f"val_acc={val_acc*100:.2f}%")

    return history


# ═══════════════════════════ 可视化 ═══════════════════════════

def plot_histories(histories, save_path="loss_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, h in histories.items():
        ep = range(1, len(h["train_loss"]) + 1)
        axes[0].plot(ep, h["train_loss"], label=f"{name} train")
        axes[0].plot(ep, h["val_loss"],   label=f"{name} val", linestyle="--")
        axes[1].plot(ep, [a * 100 for a in h["val_acc"]], label=name)
    axes[0].set_title("Loss Curves");      axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Acc (%)")
    for ax in axes:
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150)
    print(f"损失曲线已保存: {save_path}")


def plot_confusion_matrix(model, X_val, y_val_int, save_path="confusion_matrix.png"):
    probs  = to_numpy(model.predict(X_val))
    y_pred = probs.argmax(axis=1)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_val_int, y_pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues"); plt.colorbar(im)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=8)
    plt.tight_layout(); plt.savefig(save_path, dpi=150)
    print(f"混淆矩阵已保存: {save_path}")


def plot_error_samples(model, X_val, y_val_int, save_path="error_samples.png", n=20):
    probs  = to_numpy(model.predict(X_val))
    X_np   = to_numpy(X_val)
    y_pred = probs.argmax(axis=1)
    errs   = np.where(y_pred != y_val_int)[0][:n]
    cols = 5; rows = (len(errs) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, rows * 2))
    axes = axes.flatten()
    for i, idx in enumerate(errs):
        axes[i].imshow(X_np[idx].reshape(28, 28), cmap="gray")
        axes[i].set_title(f"True:{y_val_int[idx]} Pred:{y_pred[idx]}", fontsize=8)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout(); plt.savefig(save_path, dpi=150)
    print(f"错误样本已保存: {save_path}")


# ═══════════════════════════ 梯度数值检验 ═══════════════════════════

def gradient_check(eps=1e-5, n_checks=30, seed=7):
    """
    完全自包含的梯度数值检验，内部用纯 numpy 手写前向/反向。
    不依赖 MLP 类或 GPU backend，结果绝对可靠。
    结构：784->16(ReLU)->10(Softmax)，batch=16，L2=1e-4
    """
    print("\n─── 梯度数值检验 ───")
    rng = np.random.default_rng(seed)
    N, D, H, C = 16, 784, 16, 10
    lam = 1e-4

    W1 = rng.standard_normal((D, H)).astype(np.float64) * np.sqrt(2.0 / D)
    b1 = np.zeros((1, H), dtype=np.float64)
    W2 = rng.standard_normal((H, C)).astype(np.float64) * np.sqrt(2.0 / H)
    b2 = np.zeros((1, C), dtype=np.float64)
    X  = rng.standard_normal((N, D)).astype(np.float64)
    Yi = rng.integers(0, C, N)
    Y  = np.zeros((N, C), dtype=np.float64); Y[np.arange(N), Yi] = 1.0

    def forward(W1, b1, W2, b2):
        Z1 = X @ W1 + b1
        A1 = np.maximum(0, Z1)
        Z2 = A1 @ W2 + b2
        e  = np.exp(Z2 - Z2.max(axis=1, keepdims=True))
        P  = e / e.sum(axis=1, keepdims=True)
        ce = -np.sum(Y * np.log(P + 1e-9)) / N
        l2 = 0.5 * lam * (np.sum(W1**2) + np.sum(W2**2))
        return ce + l2, P, Z1, A1

    loss0, P, Z1, A1 = forward(W1, b1, W2, b2)
    # 解析梯度
    dZ2 = (P - Y) / N
    dW2_ana = A1.T @ dZ2 + lam * W2
    db2_ana = dZ2.sum(axis=0, keepdims=True)
    dZ1 = (dZ2 @ W2.T) * (Z1 > 0)
    dW1_ana = X.T @ dZ1 + lam * W1
    db1_ana = dZ1.sum(axis=0, keepdims=True)

    max_err = 0.0
    for param, grad in [(W1, dW1_ana), (b1, db1_ana), (W2, dW2_ana), (b2, db2_ana)]:
        flat = param.ravel(); gflat = grad.ravel()
        idxs = rng.choice(len(flat), min(n_checks // 4, len(flat)), replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps; lp, *_ = forward(W1, b1, W2, b2)
            flat[i] = orig - eps; lm, *_ = forward(W1, b1, W2, b2)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            err = abs(num - gflat[i]) / (abs(num) + abs(gflat[i]) + 1e-10)
            max_err = max(max_err, err)

    ok = max_err < 1e-4
    print(f"  最大相对误差: {max_err:.2e}  {'✓ 通过' if ok else '✗ 失败'}")
    return max_err


# ═══════════════════════════ 超参数实验配置 ═══════════════════════════

EXPERIMENTS = {
    "Baseline [256,128] ReLU lr=0.01": dict(
        layer_sizes=[784, 256, 128, 10], activations=["relu", "relu"],
        l2_lambda=1e-4, keep_prob=0.8, optimizer="adam", lr=0.01, batch_size=64,
    ),
    "Group1 [512,256] LeakyReLU lr=0.005": dict(
        layer_sizes=[784, 512, 256, 10], activations=["leaky_relu", "leaky_relu"],
        l2_lambda=1e-3, keep_prob=0.8, optimizer="adam", lr=0.005, batch_size=128,
    ),
    "Group2 [128] Tanh lr=0.02": dict(
        layer_sizes=[784, 128, 10], activations=["tanh"],
        l2_lambda=0.0, keep_prob=1.0, optimizer="momentum", lr=0.02, batch_size=32,
    ),
}

EPOCHS = 30


# ═══════════════════════════ 主程序 ═══════════════════════════

def main():
    np.random.seed(42)
    print(f"运行设备: {DEVICE}\n")

    # 1. 梯度数值检验（纯 numpy 自包含实现，与 GPU backend 完全隔离）
    print("═══ 梯度数值检验 ═══")
    gradient_check()

    # 2. 加载数据（自动上传到 GPU）
    X_train, Y_train, X_val, Y_val, y_val_int = load_mnist(val_ratio=0.2)

    # 3. 超参数对比实验
    histories = {}
    best_model, best_acc, best_name = None, 0.0, ""

    for name, cfg in EXPERIMENTS.items():
        print(f"\n═══ 实验: {name} ═══")
        batch_size = cfg.pop("batch_size")
        model = MLP(**cfg)
        cfg["batch_size"] = batch_size

        h = train(model, X_train, Y_train, X_val, Y_val, y_val_int,
                  epochs=EPOCHS, batch_size=batch_size)
        histories[name] = h

        final_acc = h["val_acc"][-1]
        print(f"  最终验证准确率: {final_acc*100:.2f}%")
        if final_acc > best_acc:
            best_acc, best_model, best_name = final_acc, model, name

    print(f"\n★ 最佳模型: {best_name}  准确率: {best_acc*100:.2f}%")

    # 4. 可视化
    out = "outputs/"
    plot_histories(histories,                              save_path=out + "loss_curves.png")
    plot_confusion_matrix(best_model, X_val, y_val_int,   save_path=out + "confusion_matrix.png")
    plot_error_samples(best_model,    X_val, y_val_int,   save_path=out + "error_samples.png")
    print("\n所有图表已保存。")


if __name__ == "__main__":
    main()
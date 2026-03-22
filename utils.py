"""
utils.py - 数据加载、激活函数、损失函数等辅助工具

GPU 支持：自动检测 cupy，有则用 GPU，否则回退 CPU（numpy）。
安装 cupy：pip install cupy-cuda12x   # 按 CUDA 版本选择
"""
import numpy as _np

# ═══════════════════════════ GPU / CPU 自动切换 ═══════════════════════════
try:
    import cupy as xp
    # 用 randn 测试，触发 curand，确保 DLL 完整可用
    xp.random.randn(2, 2)
    DEVICE = "GPU"
except Exception:
    import numpy as xp
    DEVICE = "CPU"

print(f"[backend] 使用 {DEVICE}（{'cupy' if DEVICE == 'GPU' else 'numpy'}）")


def to_device(arr):
    """将 numpy 数组发送到当前设备（GPU/CPU）"""
    if DEVICE == "GPU":
        return xp.asarray(arr)
    return arr


def to_numpy(arr):
    """将数组取回 CPU numpy（用于 sklearn / matplotlib）"""
    if DEVICE == "GPU":
        return xp.asnumpy(arr)
    return _np.asarray(arr)


# ═══════════════════════════ 数据加载 ═══════════════════════════

def load_mnist(val_ratio=0.2, seed=42):
    """
    加载 MNIST，归一化 + One-hot + 8:2 划分，数据自动上传到 GPU。
    """
    from sklearn.datasets import fetch_openml
    print("正在加载 MNIST（首次需联网，约需1分钟）...")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X = mnist.data.astype(_np.float32) / 255.0
    y = mnist.target.astype(_np.int32)

    rng = _np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    split = int(len(X) * (1 - val_ratio))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    Y_train = _one_hot_numpy(y_train, 10)
    Y_val   = _one_hot_numpy(y_val,   10)

    # 上传到 GPU（CPU 时是 no-op）
    X_train, Y_train = to_device(X_train), to_device(Y_train)
    X_val,   Y_val   = to_device(X_val),   to_device(Y_val)
    # y_val 保留 numpy，用于 sklearn/matplotlib
    print(f"训练集: {X_train.shape}  验证集: {X_val.shape}  设备: {DEVICE}")
    return X_train, Y_train, X_val, Y_val, y_val


def _one_hot_numpy(y, num_classes=10):
    out = _np.zeros((len(y), num_classes), dtype=_np.float32)
    out[_np.arange(len(y)), y] = 1.0
    return out


def one_hot(y, num_classes=10):
    out = xp.zeros((len(y), num_classes), dtype=xp.float32)
    out[xp.arange(len(y)), y] = 1.0
    return out


# ═══════════════════════════ 激活函数 ═══════════════════════════

def relu(z):
    return xp.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(xp.float32)


def leaky_relu(z, alpha=0.01):
    return xp.where(z > 0, z, alpha * z)

def leaky_relu_grad(z, alpha=0.01):
    return xp.where(z > 0, 1.0, alpha).astype(xp.float32)


def tanh_fn(z):
    return xp.tanh(z)

def tanh_grad(z):
    return (1.0 - xp.tanh(z) ** 2).astype(xp.float32)


def sigmoid(z):
    return 1.0 / (1.0 + xp.exp(-xp.clip(z, -500, 500)))

def sigmoid_grad(z):
    s = sigmoid(z)
    return (s * (1 - s)).astype(xp.float32)


ACTIVATIONS = {
    "relu":       (relu,       relu_grad),
    "leaky_relu": (leaky_relu, leaky_relu_grad),
    "tanh":       (tanh_fn,    tanh_grad),
    "sigmoid":    (sigmoid,    sigmoid_grad),
}


# ═══════════════════════════ Softmax & 损失 ═══════════════════════════

def softmax(z):
    e = xp.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_loss(probs, Y):
    n = len(Y)
    return -xp.sum(Y * xp.log(probs + 1e-9)) / n


# ═══════════════════════════ 参数初始化 ═══════════════════════════

def he_init(fan_in, fan_out):
    return xp.random.randn(fan_in, fan_out).astype(xp.float32) * xp.sqrt(2.0 / fan_in)

def xavier_init(fan_in, fan_out):
    return xp.random.randn(fan_in, fan_out).astype(xp.float32) * xp.sqrt(1.0 / fan_in)

def choose_init(fan_in, fan_out, activation):
    if activation in ("tanh", "sigmoid"):
        return xavier_init(fan_in, fan_out)
    return he_init(fan_in, fan_out)


# ═══════════════════════════ 辅助 ═══════════════════════════

def accuracy(probs, y_int):
    """probs 在 GPU/CPU 上，y_int 为 numpy 整数数组"""
    pred = to_numpy(probs).argmax(axis=1)
    return float(_np.mean(pred == y_int))
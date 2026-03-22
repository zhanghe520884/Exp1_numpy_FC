"""
model.py - 全连接层、Dropout 层、多层感知机（MLP）

所有张量运算通过 utils.xp 执行，GPU/CPU 透明切换。
"""
from utils import xp, to_device, ACTIVATIONS, softmax, cross_entropy_loss, choose_init


# ═══════════════════════════ 全连接层 ═══════════════════════════

class FullyConnectedLayer:
    """
    单个全连接层。
    前向传播：Z = X @ W + b，A = activation(Z)
    反向传播：链式法则计算 dW、db、dX
    """

    def __init__(self, input_dim, output_dim, activation="relu"):
        assert activation in ACTIVATIONS, f"不支持的激活函数: {activation}"
        self.act_fn, self.act_grad = ACTIVATIONS[activation]
        self.activation = activation

        self.W = None
        self.b = xp.zeros((1, output_dim), dtype=xp.float32)

        self._X = None   # 缓存输入（反向传播用）
        self._Z = None   # 缓存线性输出

        self.dW = None
        self.db = None

        # Adam 一阶/二阶矩（在 GPU 上分配）
        self.mW = xp.zeros((input_dim, output_dim), dtype=xp.float32)
        self.vW = xp.zeros((input_dim, output_dim), dtype=xp.float32)
        self.mb = xp.zeros((1, output_dim), dtype=xp.float32)
        self.vb = xp.zeros((1, output_dim), dtype=xp.float32)

    def forward(self, X, training=True):
        self._X = X
        self._Z = X @ self.W + self.b
        return self.act_fn(self._Z)

    def backward(self, dA, l2_lambda=0.0):
        """
        dA: 上游梯度（已含 /N 归一化）
        返回 dX 传递给上一层
        """
        dZ = dA * self.act_grad(self._Z)
        self.dW = self._X.T @ dZ + l2_lambda * self.W   # L2 正则梯度
        self.db = dZ.sum(axis=0, keepdims=True)
        return dZ @ self.W.T


# ═══════════════════════════ Dropout 层 ═══════════════════════════

class DropoutLayer:
    """Inverted Dropout：训练时随机置零并缩放，推理时原样输出。"""

    def __init__(self, keep_prob=0.8):
        self.keep_prob = keep_prob
        self._mask = None

    def forward(self, X, training=True):
        if training and self.keep_prob < 1.0:
            self._mask = (xp.random.rand(*X.shape) < self.keep_prob).astype(xp.float32)
            return X * self._mask / self.keep_prob
        return X

    def backward(self, dA, **kwargs):
        if self._mask is not None:
            return dA * self._mask / self.keep_prob
        return dA


# ═══════════════════════════ 多层感知机 ═══════════════════════════

class MLP:
    """
    多层感知机，支持：
    - 任意隐藏层结构
    - ReLU / LeakyReLU / Tanh / Sigmoid 激活函数
    - L2 正则化 + Dropout
    - SGD / Momentum / Adam 优化器
    - He / Xavier 参数初始化（按激活函数自动选择）
    - GPU/CPU 透明运行
    """

    def __init__(
        self,
        layer_sizes,
        activations,
        l2_lambda=1e-4,
        keep_prob=1.0,
        optimizer="adam",
        lr=0.01,
        momentum=0.9,
        beta1=0.9, beta2=0.999, eps=1e-8,
    ):
        assert len(activations) == len(layer_sizes) - 2, \
            "activations 数量应等于隐藏层数量（不含输出层）"

        self.l2_lambda   = l2_lambda
        self.optimizer   = optimizer
        self.lr          = lr
        self.momentum_coef = momentum
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0   # Adam 时间步

        # ── 构建隐藏层 ──
        self.fc_layers:      list[FullyConnectedLayer] = []
        self.dropout_layers: list[DropoutLayer]        = []

        for i in range(len(layer_sizes) - 2):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            act = activations[i]
            layer = FullyConnectedLayer(fan_in, fan_out, activation=act)
            layer.W = choose_init(fan_in, fan_out, act)
            self.fc_layers.append(layer)
            self.dropout_layers.append(DropoutLayer(keep_prob))

        # ── 输出层（线性激活，Softmax 在损失里处理）──
        out = FullyConnectedLayer(layer_sizes[-2], layer_sizes[-1], activation="relu")
        out.W = choose_init(layer_sizes[-2], layer_sizes[-1], "relu") * 0.1
        out.act_fn   = lambda z: z
        out.act_grad = lambda z: xp.ones_like(z)  # xp 已在模块顶部导入，GPU/CPU 均正确
        self.fc_layers.append(out)

        # Momentum 速度缓存
        self._vel_W = [xp.zeros_like(l.W) for l in self.fc_layers]
        self._vel_b = [xp.zeros_like(l.b) for l in self.fc_layers]

    # ── 前向传播 ──
    def forward(self, X, training=True):
        A = X
        for i, fc in enumerate(self.fc_layers[:-1]):
            A = fc.forward(A, training)
            A = self.dropout_layers[i].forward(A, training)
        return self.fc_layers[-1].forward(A, training)   # logits

    # ── 损失计算 ──
    def loss(self, X, Y, training=False):
        logits = self.forward(X, training)
        probs  = softmax(logits)
        ce     = cross_entropy_loss(probs, Y)
        l2     = sum(xp.sum(l.W ** 2) for l in self.fc_layers)
        return ce + 0.5 * self.l2_lambda * l2, probs

    # ── 反向传播 ──
    def backward(self, probs, Y):
        """
        输出层：Softmax + CrossEntropy 合并化简得 dZ = (probs - Y) / N
        """
        N  = len(Y)
        dA = (probs - Y) / N   # 输出层梯度

        dA = self.fc_layers[-1].backward(dA, self.l2_lambda)
        for i in reversed(range(len(self.fc_layers) - 1)):
            dA = self.dropout_layers[i].backward(dA)
            dA = self.fc_layers[i].backward(dA, self.l2_lambda)

    # ── 参数更新 ──
    def update(self):
        self.t += 1
        for i, layer in enumerate(self.fc_layers):

            if self.optimizer == "sgd":
                layer.W -= self.lr * layer.dW
                layer.b -= self.lr * layer.db

            elif self.optimizer == "momentum":
                self._vel_W[i] = self.momentum_coef * self._vel_W[i] + self.lr * layer.dW
                self._vel_b[i] = self.momentum_coef * self._vel_b[i] + self.lr * layer.db
                layer.W -= self._vel_W[i]
                layer.b -= self._vel_b[i]

            elif self.optimizer == "adam":
                # 更新一阶矩
                layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
                layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db
                # 更新二阶矩
                layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * layer.dW ** 2
                layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * layer.db ** 2
                # 偏差修正
                mW_hat = layer.mW / (1 - self.beta1 ** self.t)
                mb_hat = layer.mb / (1 - self.beta1 ** self.t)
                vW_hat = layer.vW / (1 - self.beta2 ** self.t)
                vb_hat = layer.vb / (1 - self.beta2 ** self.t)
                # 参数更新
                layer.W -= self.lr * mW_hat / (xp.sqrt(vW_hat) + self.eps)
                layer.b -= self.lr * mb_hat / (xp.sqrt(vb_hat) + self.eps)

    # ── 预测 ──
    def predict(self, X):
        return softmax(self.forward(X, training=False))
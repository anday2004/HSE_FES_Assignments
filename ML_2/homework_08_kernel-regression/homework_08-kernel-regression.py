import numpy as np
from sklearn.base import RegressorMixin
from sklearn.gaussian_process.kernels import RBF
from tqdm import tqdm


class KernelRidgeRegression(RegressorMixin):
    """
    Kernel Ridge regression class
    """

    def __init__(self,         
        lr=0.01,
        regularization=0.1,
        tolerance=1e-3,
        max_iter=1000,
        batch_size=64,
        kernel_scale=1,
        method="gradient"
    ):
        
        self.lr: float = lr
        self.regularization: float = regularization
        self.w: np.ndarray | None = None
        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.batch_size: int = batch_size
        self.loss_history: list[float] = []
        self.kernel = RBF(kernel_scale)
        self.method = method

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:

        return 0.5 * np.linalg.norm(self.kernel_ @ self.w + self.b - y) + 0.5 * self.regularization * self.w.T @ self.kernel_ @ self.w 

    def calc_grad(self, x: np.ndarray, y: np.ndarray) -> float:

        batch_indices = np.random.randint(0, x.shape[0], self.batch_size)
        
        kernel_batch = self.kernel(x[batch_indices], x)
        return kernel_batch.T @ kernel_batch @ self.w - kernel_batch.T @ np.ones((self.batch_size, 1)) * 1 +  kernel_batch.T @ y[batch_indices] + self.regularization * self.kernel_ @ self.w, np.mean(self.b + kernel_batch @ self.w - y[batch_indices])
    
    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_grad(x, y))
    
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:

        weight_difference = -self.lr * gradient[0]
        self.w = self.w + weight_difference[0]
        self.b = self.b - self.lr * gradient[1]
        return weight_difference
    
    def fit(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        if self.method == "gradient":
            return self.fit_gradient(x, y)
        elif self.method == "analytical":
            return self.fit_analytical(x, y)

    def fit_gradient(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":

        self.w = np.random.rand(x.shape[0]).reshape(x.shape[0], 1)
        self.b = np.random.rand(1).reshape(-1, 1)
        self.kernel_ = self.kernel(x)
        for _ in tqdm(range(self.max_iter)):
            self.loss_history.append(self.calc_loss(x, y).item())
            delta = self.step(x, y)
            if np.linalg.norm(delta)**2  < self.tolerance:
                break

        self.loss_history.append(self.calc_loss(x, y).item())
        return self
    
    def fit_analytical(self, x: np.ndarray, y: np.ndarray) -> "KernelRidgeRegression":
        self.kernel_ = self.kernel(x)

        n = x.shape[0]
        ones = np.ones((n, 1))

        # Решаем систему уравнений для b
        b_numerator = ones.T @ y
        b_denominator = ones.T @ ones
        self.b = (b_numerator / b_denominator).item()  # Преобразуем в скаляр

        # Регуляризованная матрица для w
        regularized_kernel = self.kernel_.T @ self.kernel_ + self.regularization * self.kernel_

        # Целевой вектор для w
        target = self.kernel_.T @ (y - ones * self.b)

        # Решаем систему уравнений для w
        self.w = np.linalg.solve(regularized_kernel, target)

        return self

    def predict(self, x_train, x_test: np.ndarray) -> np.ndarray:

        return self.kernel(x_train, x_test).T @ self.w + self.b
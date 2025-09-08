
from Linear_svm import *
class GaussianSVM(LinearSVM):
    def __init__(self, class_number, C=1, cigma=4):
        # Call parent constructor for common attributes
        super().__init__(class_number, C)
        self.cigma = cigma
        self.kernelmatrix = None

    def set_training_data(self, X,Y):
        self.train_features_tensor = X #save for later check if model has been fitted
        self.train_labels = torch.where(Y == self.class_number, 1.0, -1.0)
        self.n_samples = len(X)
        self.kernelmatrix = self.gaussian_kernel(self.train_features_tensor)#precompute this to avoid doing it in training phase
        #initalize weights to zero
        if self.W is None or self.b is None:
            self.W = torch.zeros(self.n_samples).requires_grad_()
            self.b = torch.zeros(1).requires_grad_()
    def predict(self, X):
        if X.shape == self.train_features_tensor.shape and torch.allclose(X, self.train_features_tensor, atol=1e-6):
            kernel = self.kernelmatrix  # Reuse precomputed kernel
        else:
            kernel = self.gaussian_kernel(X)
        return kernel @ self.W + self.b
    def loss(self, y_true, y_pred, train_loss=True):
        hinge = torch.mean(self.cost(y_true, y_pred))
        if not train_loss:
            return hinge
        regularization = 0.5 * (self.W @ (self.kernelmatrix @ self.W))
        return self.C * hinge + regularization

    def train(self, X, Y):
        #Method for training model using Quadratic Programming
        if (self.train_features_tensor is None or
                self.train_labels is None or
                self.kernelmatrix is None):
            raise RuntimeError("Call set_training_data(X, Y) before train() for GaussianSVM.")


        y_np = self.train_labels.detach().cpu().double().numpy().reshape(-1, 1)  # (n,1)
        K = self.kernelmatrix.detach().cpu().double().numpy()                     # (n,n)
        n = K.shape[0]
        C = float(self.C)

        # Dual QP
        P = (y_np @ y_np.T) * K
        q = -np.ones(n, dtype=float)
        G = np.vstack([np.eye(n), -np.eye(n)])
        h = np.hstack([C * np.ones(n), np.zeros(n)])
        A = y_np.reshape(1, -1)
        b_eq = np.array([0.0], dtype=float)

        solvers.options["show_progress"] = True
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b_eq))
        a = np.array(sol["x"]).reshape(-1)

        sv_mask = a > 1e-6
        if not np.any(sv_mask):
            self.W = torch.zeros(n)
            self.b = torch.zeros(1)
            self.is_fitted = True
            print("Training completed via QP (degenerate RBF solution).")
            return

        # Dual coefficients (alpha_i * y_i)
        coeffs = a * y_np.reshape(-1)
        f_all = K.T @ coeffs

        margin_sv = (a > 1e-6) & (a < C - 1e-6)
        idx = np.where(margin_sv)[0] if np.any(margin_sv) else np.where(sv_mask)[0]
        b_vals = y_np.reshape(-1)[idx] - f_all[idx]
        b = float(b_vals.mean())

        self.W = torch.from_numpy(coeffs.astype(np.float32))  # used by predict: K @ W + b
        self.b = torch.tensor([b], dtype=torch.float32)
        self.is_fitted = True
        print("Training completed via QP (cvxopt, RBF).")
    def gaussian_kernel(self, X):#Kernel Trick
        dist_squared = torch.cdist(X, self.train_features_tensor) ** 2
        kernel = torch.exp(-dist_squared / (2 * self.cigma ** 2))
        return kernel

    def get_hyperparams(self):
        return f'C={self.C},cigma={self.cigma}'

    # No need to override cost(), loss(), or gradientDescent() - they work perfectly via inheritance!

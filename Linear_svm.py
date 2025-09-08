import numpy as np
import torch
from cvxopt import matrix, solvers
class LinearSVM:
    def __init__(self, class_number, C=1):
        self.class_number = class_number
        self.C = C
        self.W = None
        self.b = None
        self.train_features_tensor = None  # Consistent naming
        self.train_labels = None
        self.is_fitted = False

    def set_training_data(self, X, Y):
        self.train_features_tensor = X  #save for later check if model has been fitted
        self.train_labels = torch.where(Y == self.class_number, 1.0, -1.0)
        self.num_of_features = X.shape[1]#n
        if self.W is None or self.b is None:#initalize weights to zero
            self.W = torch.zeros(self.num_of_features).requires_grad_()
            self.b = torch.zeros(1).requires_grad_()

    def predict(self, X):
        return X @ self.W + self.b
    def train(self, X, Y):
        #Method for training using Quadratic Programming.
        if self.train_features_tensor is None or self.train_labels is None:
            raise RuntimeError("Call set_training_data(X, Y) before train().")

        X_np = self.train_features_tensor.detach().cpu().double().numpy()   # (n,d)
        y_np = self.train_labels.detach().cpu().double().numpy().reshape(-1, 1)  # (n,1)
        n, d = X_np.shape
        C = float(self.C)

        # Dual QP for linear SVM
        K = X_np @ X_np.T
        P = (y_np @ y_np.T) * K
        q = -np.ones(n, dtype=float)
        G = np.vstack([np.eye(n), -np.eye(n)])
        h = np.hstack([C * np.ones(n), np.zeros(n)])
        A = y_np.reshape(1, -1)
        b_eq = np.array([0.0], dtype=float)

        solvers.options["show_progress"] = True
        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b_eq))
        a = np.array(sol["x"]).reshape(-1)

        # Build W, b
        sv_mask = a > 1e-6
        if not np.any(sv_mask):
            self.W = torch.zeros(d)
            self.b = torch.zeros(1)
            self.is_fitted = True
            print("Training completed via QP (degenerate linear solution).")
            return

        Wy = (a[sv_mask] * y_np[sv_mask].reshape(-1))[:, None]
        W = (Wy * X_np[sv_mask]).sum(axis=0)

        margin_sv = (a > 1e-6) & (a < C - 1e-6)
        if np.any(margin_sv):
            b_vals = y_np[margin_sv].reshape(-1) - (X_np[margin_sv] @ W)
        else:
            b_vals = y_np[sv_mask].reshape(-1) - (X_np[sv_mask] @ W)
        b = float(b_vals.mean())

        self.W = torch.from_numpy(W.astype(np.float32))
        self.b = torch.tensor([b], dtype=torch.float32)
        self.is_fitted = True
        print("Training completed via QP (cvxopt, linear).")
    def cost(self, y_true, y_pred):
        return torch.clamp(1 - y_true * y_pred, min=0)#hinge loss

    def loss(self, y_true, y_pred, train_loss=True):
        if train_loss:#loss for training (regularization added)
            cost_term = torch.mean(self.cost(y_true, y_pred))
            reg_term = 0.5 * (self.W ** 2).sum()
            return self.C * cost_term + reg_term
        else:
             # cost for evaluation of performance (unregulated
             return torch.mean(self.cost(y_true, y_pred))

    def gradientDescent(self, X, Y, alpha, epochs, batch_size=32):
        #Method for training model using Gradient Descent
        y_true = torch.where(Y == self.class_number, 1.0, -1.0)
        n_samples = X.shape[0]
        prev_loss = None

        for i in range(epochs):
            if batch_size == None:
                batch_X=X
                batch_y=y_true
            else:
                start_idx = torch.randint(0, n_samples - batch_size + 1, (1,)).item()
                batch_X = X[start_idx:start_idx + batch_size]
                batch_y = y_true[start_idx:start_idx + batch_size]

            y_predict = self.predict(batch_X)  # Calls child's predict() via polymorphism
            loss = self.loss(y_true=batch_y, y_pred=y_predict)
            loss.backward()
            with torch.no_grad():
                self.W -= alpha * self.W.grad
                self.b -= alpha * self.b.grad
                self.W.grad.zero_()
                self.b.grad.zero_()

            loss_val = loss.item()
            if prev_loss is not None:
                diff = abs(prev_loss - loss_val)
                if diff < 1e-6:
                    break
            prev_loss = loss_val

            if i % 1000 == 0 or i == epochs - 1:
                print(f"Iter {i}, Alpha: {alpha}, Loss: {loss.item():.5f}")
        if torch.equal(X, self.train_features_tensor) and torch.equal(y_true, self.train_labels):
            self.is_fitted = True#this is a check if model has ben fitted on the tra

    def get_hyperparams(self):
        return f'C={self.C}'

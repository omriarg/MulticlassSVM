
from Gaussian_svm import GaussianSVM
from Linear_svm import LinearSVM
from Polynomial_svm import Polynomial_SVM
from Data_initalization_and_scaling import data_init
import torch

class MultiClassSVM():
    #This is a wrapper class from MultiClass (One-vs-All) SVM
    #it offers encapsulation of the model behaviour, train,predict,prepare data.
    def __init__(self, raw_data_df=None):
        if raw_data_df is not None:
            self.set_data(raw_data_df)

    def set_data(self, data):#the DataSet to be proccesed.
        self.data_processed = data_init(data)
        self.training_set, self.cross_eval_set, self.test_set = self.data_processed.get_splits()
        self.models = []  # Model per class

    def evaluate_loss_on_data(self, X, Y, dataset_type=''):
        #evaluate and report the average loss of the models on given input data.
        with torch.no_grad():
            losses = []
            for model in self.models:
                scores = model.predict(X)
                y_true_pm1 = torch.where(
                    Y == model.class_number,
                    torch.ones_like(scores, dtype=scores.dtype, device=scores.device),
                    -torch.ones_like(scores, dtype=scores.dtype, device=scores.device)
                )
                val_loss = model.loss(y_true_pm1, scores, train_loss=False)  # unregularized
                losses.append(val_loss if torch.is_tensor(val_loss) else torch.tensor(val_loss, device=scores.device))
            mean_loss = torch.stack(losses).mean().item()
            print(f"loss on {dataset_type} set: {mean_loss:.6f} (one-vs-rest mean)")
            return mean_loss

    def test_best_kernel_and_hyperparams(self, method='QP'):
        # ---grids---
        rbf_C_powers   = (-5, -1, 1, 2, 4)
        rbf_gamma_pows = (-9,  -5, -3, -1)
        rbf_C = [2.0**p for p in rbf_C_powers]
        rbf_gamma = [2.0**p for p in rbf_gamma_pows]
        rbf_sigmas = [(1.0 / (2.0 * g)) ** 0.5 for g in rbf_gamma]
        gaussian_grid = [{'C': c, 'cigma': s} for c in rbf_C for s in rbf_sigmas]

        poly_C       = [0.3, 1.0, 1.7, 2.5, 8.0]
        poly_degrees = [2, 3, 4]
        poly_consts  = [0.0, 1.0]
        polynomial_grid = [{'C': c, 'degree': d, 'const': r}
                           for c in poly_C for d in poly_degrees for r in poly_consts]

        results_by_kernel = {}
        summaries = {}

        bp_rbf, bcv_rbf, bacc_rbf, rows_rbf = self._test_best_parameters('gaussian', gaussian_grid, method=method)
        results_by_kernel['gaussian'] = rows_rbf
        summaries['gaussian'] = {'best_params': bp_rbf, 'cv_loss': bcv_rbf, 'cv_acc': bacc_rbf}
    
        bp_poly, bcv_poly, bacc_poly, rows_poly = self._test_best_parameters('polynomial', polynomial_grid, method=method)
        results_by_kernel['polynomial'] = rows_poly
        summaries['polynomial'] = {'best_params': bp_poly, 'cv_loss': bcv_poly, 'cv_acc': bacc_poly}
    
        best_kernel = None
        best_cv_loss = float('inf')
        best_cv_acc  = -1.0
        for k, s in summaries.items():
            if (s['cv_loss'] < best_cv_loss) or (s['cv_loss'] == best_cv_loss and s['cv_acc'] > best_cv_acc):
                best_kernel = k
                best_cv_loss = s['cv_loss']
                best_cv_acc  = s['cv_acc']
    
        best_params = summaries[best_kernel]['best_params']
        print("\n--- Summary (by CV loss; tie-break CV acc) ---")
        for k, s in summaries.items():
            print(f"{k:10s} | CV loss={s['cv_loss']:.6f} | CV acc={s['cv_acc']:.4f} | Params={s['best_params']}")
        print(f"\nBest overall: kernel={best_kernel}, params={best_params}, CV loss={best_cv_loss:.6f}, CV acc={best_cv_acc:.4f}")
    
        # keep the best model.
        params_to_store = dict(best_params) if best_params is not None else {}
        params_to_store = {k: v for k, v in params_to_store.items() if not (isinstance(k, str) and k.startswith('_'))}
        params_to_store['kernel_type'] = best_kernel
        self.model_params = params_to_store

    
        return best_kernel, best_params, best_cv_loss, best_cv_acc, results_by_kernel
    def _test_best_parameters(self, kernel_type, params_grid, method='QP'):
        X_train, Y_train = self.training_set
        X_cv, Y_cv       = self.cross_eval_set
    
        use_gd = (str(method).upper() == 'GD')#separate paths for training with Graident Descent,and Quadratic programming
        alpha_list = [1e-8, 1e-6, 1e-3] if use_gd else None
    
        best_cv_loss = float('inf')
        best_cv_acc  = -1.0
        best_params  = None
        best_rows_by_C = {}
    
        print(f"\n--- Hyperparameter Search for {kernel_type} via {method} (by CV loss, tie-break CV acc) ---")
        for params in params_grid:
            C_val = float(params.get('C', 1.0))
    
            if use_gd:
                # -------- GD path: loop over alphas --------
                for alpha in alpha_list:
                    to_log = dict(params); to_log['alpha'] = alpha
                    print(f"Testing params: {to_log}")

                    self.create_classifiers(kernel_type=kernel_type, params=params, num_classes=3)
                    self.train(X_train, Y_train, method='GD', alpha=alpha, epochs=10000, batch_size=32)
    
                    train_loss = float(self.evaluate_loss_on_data(X_train, Y_train, dataset_type='train'))
                    cv_loss    = float(self.evaluate_loss_on_data(X_cv,    Y_cv,    dataset_type='cross_eval'))
                    train_acc  = float(self.evaluate_accuracy_on_data(X_train, Y_train, dataset_type='train'))
                    cv_acc     = float(self.evaluate_accuracy_on_data(X_cv, Y_cv, dataset_type='cross_eval'))
    
                    if (cv_loss < best_cv_loss) or (cv_loss == best_cv_loss and cv_acc > best_cv_acc):
                        best_cv_loss = cv_loss
                        best_cv_acc  = cv_acc
                        best_params  = dict(params)
                        best_params['alpha'] = alpha
                        best_params['_method'] = 'GD'
    
                    prev = best_rows_by_C.get(C_val)
                    row  = {
                        'C': C_val, 'train_loss': train_loss, 'cv_loss': cv_loss,
                        'train_acc': train_acc, 'cv_acc': cv_acc, 'params': dict(params),
                        'alpha': alpha, '_method': 'GD'
                    }
                    if (prev is None) or (cv_loss < prev['cv_loss']) or (cv_loss == prev['cv_loss'] and cv_acc > prev['cv_acc']):
                        best_rows_by_C[C_val] = row
    
            else:
                # -------- QP path: single run per params (NO alpha loop) --------
                print(f"Testing params: {params}")
    
                self.create_classifiers(kernel_type=kernel_type, params=params, num_classes=3)
                self.train(X_train, Y_train, method='QP')
    
                train_loss = float(self.evaluate_loss_on_data(X_train, Y_train, dataset_type='train'))
                cv_loss    = float(self.evaluate_loss_on_data(X_cv,    Y_cv,    dataset_type='cross_eval'))
                train_acc  = float(self.evaluate_accuracy_on_data(X_train, Y_train, dataset_type='train'))
                cv_acc     = float(self.evaluate_accuracy_on_data(X_cv, Y_cv, dataset_type='cross_eval'))
    
                if (cv_loss < best_cv_loss) or (cv_loss == best_cv_loss and cv_acc > best_cv_acc):
                    best_cv_loss = cv_loss
                    best_cv_acc  = cv_acc
                    best_params  = dict(params)
                    best_params['_method'] = 'QP'
    
                prev = best_rows_by_C.get(C_val)
                row  = {
                    'C': C_val, 'train_loss': train_loss, 'cv_loss': cv_loss,
                    'train_acc': train_acc, 'cv_acc': cv_acc, 'params': dict(params),
                    '_method': 'QP'
                }
                if (prev is None) or (cv_loss < prev['cv_loss']) or (cv_loss == prev['cv_loss'] and cv_acc > prev['cv_acc']):
                    best_rows_by_C[C_val] = row
    
        per_C_rows = sorted(best_rows_by_C.values(), key=lambda r: r['C'])
        print(f"\nBest for {kernel_type}: params={best_params}, CV loss={best_cv_loss:.6f}, CV acc={best_cv_acc:.4f}")
        return best_params, best_cv_loss, best_cv_acc, per_C_rows
    
    def create_classifiers(self, kernel_type, params, num_classes=3):
        self.models = []
        X_train,Y_train=self.training_set
        for class_number in range(num_classes):
            if kernel_type == 'linear':
                model = LinearSVM(class_number=class_number, C=params.get('C', 1))
            elif kernel_type == 'polynomial':
                model = Polynomial_SVM(class_number=class_number, C=params.get('C', 1),
                                       degree=params.get('degree', 2),
                                       const=params.get('const', 0))
            else:
                model = GaussianSVM(class_number=class_number, C=params.get('C', 1),
                                    cigma=params.get('cigma', 1))

            model.set_training_data(X_train,Y_train)  ##this is the base data model will be trained on
            self.models.append(model)
    def train(self, X, Y, alpha=0.001, epochs=40000, batch_size=32,method='GD'):
        if not self.models:
            raise RuntimeError("Models not initialized. Call create_classifiers() before train().")
        for model in self.models:
            if method=='QP':
                model.train(X,Y)
            else:#method is Gradient Descent
                model.gradientDescent(X, Y, alpha=alpha, epochs=epochs, batch_size=batch_size)
    def prepare_new_data(self,data_df):
        return self.data_processed.prepare_new_df(data_df)
    def predict(self, X):
        with torch.no_grad():
            # margins from each OvR classifier: [num_classes, N]
            all_preds = [m.predict(X).view(-1) for m in self.models]
            logits = torch.stack(all_preds)  # [C, N]

            # softmax
            logits = logits - logits.max(dim=0, keepdim=True).values
            exp_scores = torch.exp(logits)
            probs = exp_scores / exp_scores.sum(dim=0, keepdim=True).clamp_min(1e-12)  # [C, N]

            return probs.argmax(dim=0)  #returns class indices

    def evaluate_accuracy_on_data(self, X, Y, dataset_type=''):
        #evaluate the average accuracy of the models on input Data.
        preds = self.predict(X)
        correct = (preds == Y).sum().item()
        total = len(Y)
        accuracy = correct / total
        print(f"Accuracy on {dataset_type} set: {accuracy * 100:.2f}% ({correct}/{total} correct)")
        return accuracy


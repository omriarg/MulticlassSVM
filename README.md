Multiclass SVM for Stock Price Movement Prediction
==================================================

This repository contains the implementation of a modular Multi-Class Support Vector Machine (SVM) framework in PyTorch.  
The project was developed as part of the "Machine Learning" course and focuses on predicting stock price movements using technical indicators and SVM models with different kernels.
This was implemented from scratch from educational purposes, without the use of Libraries outside of Mathematical operations.

--------------------------------------------------
🚀 Project Overview
--------------------------------------------------
The goal of this project is to design, implement, and evaluate SVM models for multi-class classification of stock price movements.  
We classify daily stock returns into three categories (e.g., drop / stable / rise), ensuring balanced class distributions via quantile-based labeling.

The system supports multiple kernelized SVMs:
- Linear SVM – baseline linear decision boundary
- Polynomial SVM – captures polynomial feature interactions
- Gaussian (RBF) SVM – models complex non-linear decision boundaries

--------------------------------------------------
🧮 Feature Engineering
--------------------------------------------------
We compute common technical indicators from historical stock price data (fetched via Yahoo Finance):

- SMA (Simple Moving Average) – tracks short- and long-term trends
- EMA (Exponential Moving Average) – gives more weight to recent prices
- RSI (Relative Strength Index) – momentum indicator for overbought/oversold states
- Volatility (rolling std) – captures price variability
- Momentum / Rate of Change (ROC) – measures speed of price movements
- On-Balance Volume (OBV) – combines price & volume for demand tracking
- Lag features – past returns as predictive signals

Data Leakage Prevention:
- Features that depend on future information (e.g., today’s Close when predicting today’s movement) are excluded.  
- Labels are generated using quantile thresholds, ensuring each class has roughly equal representation.

--------------------------------------------------
⚙️ Implementation
--------------------------------------------------
- Core SVMs are implemented from scratch in PyTorch, with:
  * Gradient Descent–based primal optimization
  * Quadratic Programming (QP) dual optimization (for kernelized SVMs)

- MultiClassSVM wrapper uses a one-vs-rest strategy to extend binary SVMs to multiple classes.

- Loss Function: Hinge loss with regularization term:
  
**L(w) = C * (1/n) * Σ max(0, 1 − yᵢ * f(xᵢ)) + 0.5 * (wᵀ K w)**

Where:  
- `yᵢ ∈ {−1, +1}` – true label for sample *i* (per one-vs-rest setup)  
- `f(xᵢ)` – predicted score for sample *i*  
- `K` – kernel matrix (Linear, Polynomial, or Gaussian/RBF)  
- `C > 0` – regularization parameter  
- `n` – number of training samples  

--------------------------------------------------
📊 Evaluation
--------------------------------------------------
- Dataset: Stock data from Yahoo Finance (e.g., AAPL, GOOGL, NVDA)
- Data split: 70% train / 15% validation / 15% test
- Metrics:
  * Macro Precision, Recall, F1 – balanced performance across classes
  * Micro Precision, Recall, F1 – overall accuracy across samples
  * Confusion Matrices for per-class breakdown

--------------------------------------------------
📂 Project Structure
--------------------------------------------------
│
├── Data_initialization_and_scaling.py   # Feature engineering and dataset preparation (with leakage prevention)

├── Linear_svm.py                        # Base SVM class (primal & kernelized loss)

├── Polynomial_svm.py                    # Polynomial kernel SVM

├── Gaussian_svm.py                      # Gaussian (RBF) kernel SVM

├── Svm_Main.py                          # Main pipeline – training, prediction, evaluation

├── requirements.txt                     # Dependencies

└── README.md                            # Project documentation

--------------------------------------------------
🛠️ How to Run
--------------------------------------------------
1. Install dependencies:
   pip install -r requirements.txt

2. Run pipeline (example with Google stock, 5-year daily data, 3 classes):
   python pipeline.py --ticker GOOGL --period 5y --interval 1d --num_classes 3

3. Check results:
   - Classification reports & confusion matrices are printed
   - Processed data and model artifacts are saved in the `Stocks/` directory

--------------------------------------------------
📖 Key Learnings
--------------------------------------------------
- SVMs can model linear and non-linear decision boundaries using kernels
- Regularization (C, kernel parameters) balances overfitting vs underfitting
- Quantile-based labeling ensures balanced class distributions
- QP vs Gradient Descent highlights the trade-off between exact solutions and scalability

--------------------------------------------------
📚 References
--------------------------------------------------
- Vapnik, V. (1995). The Nature of Statistical Learning Theory
- Cortes, C., & Vapnik, V. (1995). Support-Vector Networks

- Hastie, Tibshirani, Friedman – The Elements of Statistical Learning
- Kuhn, M., & Johnson, V. – Applied Predictive Modeling

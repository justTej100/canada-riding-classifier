# Training Classifiers on Canadian Political Data

A web version of a machine learning report that predicts which of Canada's 343 federal ridings elected a Liberal MP in the 2025 federal election. It trains a soft-margin support vector machine on census and demographic data and compares three optimization algorithms across three kernel functions and two feature setups.

170 ridings elected a Liberal MP and 173 did not so the classes are nearly balanced. Each riding is described by 16 numeric census features grouped into demographic, education, economic, housing and language and transport themes.

The report is written in Python. AdaDelta is implemented from scratch in NumPy while simulated annealing and BFGS use SciPy and cross-validation uses scikit-learn.

## The report

1. **Introduction and Problem Overview** states the problem and the objectives and defines the soft-margin SVM. The loss is hinge loss with L2 regularization. BFGS needs a smooth gradient so it uses a Huber-style smoothed hinge with delta set to 0.5.
2. **Data Processing** explores the class balance and the feature distributions. It standardizes every feature and applies PCA. Eight components keep 98.2 percent of the variance. The first component reads as wealth and education and the second as French speaker rate and age. The third reads as unemployment and housing need and the fourth as population and cycling.
3. **Modelling** introduces the linear, RBF and polynomial kernels. Each optimizer gets a plain explanation and its update rule. Pseudocode and the Python implementation follow. Kernel settings are tuned with 5-fold stratified grid search.
4. **Experiments** runs all nine optimizer and kernel combinations twice. Experiment A uses the 16 raw standardized features and Experiment B uses the 8 PCA features. Both use the same stratified 80/20 split with 274 training ridings and 69 test ridings. Charts compare accuracy, loss convergence and runtime. A delta table shows how PCA changed each result.
5. **Conclusion** sums up what PCA did, how each optimizer behaved and which model won.

## Key results

Test accuracy in percent.

| Kernel | Optimizer | Raw features | PCA features |
|---|---|---|---|
| Linear | Simulated Annealing | 68.1 | 69.6 |
| Linear | BFGS | 68.1 | 68.1 |
| Linear | AdaDelta | 68.1 | 68.1 |
| RBF | Simulated Annealing | 55.1 | 56.5 |
| RBF | BFGS | 62.3 | 66.7 |
| RBF | AdaDelta | 69.6 | 68.1 |
| Polynomial | Simulated Annealing | 72.5 | 71.0 |
| Polynomial | BFGS | 71.0 | 71.0 |
| Polynomial | AdaDelta | 49.3 | 63.8 |

- **Best model.** Simulated annealing with the polynomial kernel on raw features reached 72.5 percent test accuracy. On the 69 test ridings precision and recall both sit between 0.71 and 0.74 and the F1 score is 0.72 for each class.
- **Tuning.** Grid search chose C of 1 and gamma of 0.1 for the RBF kernel with a CV accuracy of 79.3 percent. For the polynomial kernel it chose degree 4 with coef0 of 5 and C of 0.01 with a CV accuracy of 78.7 percent.
- **PCA had mixed effects.** It helped in 4 of 9 combinations and hurt in 2. It made no difference in 3. The biggest gain was AdaDelta with the polynomial kernel which rose from 49.3 to 63.8 percent. BFGS with the RBF kernel gained 4.3 points. Both drops were 1.4 points.
- **Optimizers.** BFGS was the steadiest and fastest and never took more than about a second. Simulated annealing was the slowest at about 25 seconds on the RBF and polynomial kernels. AdaDelta was the most sensitive to the choice of feature representation.
- **Reading the numbers.** The test set has only 69 ridings so a single riding is worth about 1.4 points. Small gaps between runs should be read with that in mind.
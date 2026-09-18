# Training Classifiers on Canadian Political Data

A comparative study of optimization algorithms for soft-margin SVM classification.

This repository is the web version of a machine learning report. The report trains a soft-margin support vector machine to predict which of Canada's 343 federal ridings elected a Liberal MP in the 2025 federal election. It uses census and demographic data for each riding. The main goal is to compare three optimization algorithms under three kernel functions and two feature representations.

The site is a single page built from the original Jupyter notebook. The written explanations, equations, code and plots are all included.

## The data

Each row is one riding. 170 ridings elected a Liberal MP and 173 did not. The classes are nearly balanced so plain accuracy is a fair way to score the models. Liberal is labelled +1 and every other result is labelled -1 which is the standard label convention for an SVM.

Each riding has 16 numeric features.

- Demographic features cover average age overall and for men and women separately.
- Education features cover the share of people without post-secondary education with the same three splits.
- Economic features cover median household income and the unemployment rate with the same three splits.
- Housing is the share of households in core housing need.
- The last three are the French speaker rate, the cycling commute rate and total population.

All features are standardized before modelling. The dataset file is not part of this repository. Only the report is.

## What is in the report

**1. Introduction and Problem Overview.** This section states the problem and the objectives and defines the soft-margin SVM. The loss is hinge loss with L2 regularization. Hinge loss has a kink that gradient methods cannot handle so BFGS uses a smooth Huber-style version with delta set to 0.5. Simulated annealing needs no gradient so it uses the original loss.

**2. Data Processing.** This section explores the class balance and compares feature distributions for Liberal and non-Liberal ridings. It then standardizes the features and runs PCA. Eight components keep 98.2 percent of the variance. A scree plot and a loadings heatmap show what the first components capture. The report reads the first as wealth and education and the second as French speaker rate and age. It reads the third as unemployment and housing need and the fourth as population and cycling.

**3. Modelling.** This section covers the three kernels, which are linear, RBF and polynomial. It then walks through the three optimizers, which are simulated annealing, BFGS and AdaDelta. Each one gets a plain explanation and its update rule. Each also gets pseudocode and a table of tuning parameters. The Python code for each one follows. A comparison table sets them side by side. The last part tunes the kernels with 5-fold stratified grid search using scikit-learn's SVC as a fast stand-in for the custom optimizers. The RBF grid had 25 settings and the best CV accuracy was 79.3 percent with C of 1 and gamma of 0.1. The polynomial grid had 45 settings and the best CV accuracy was 78.7 percent with C of 0.01, degree 4 and coef0 of 5. The optimizers themselves were not tuned separately because that would take too long.

**4. Running the Experiments.** Every optimizer is run with every kernel in two parallel experiments. Experiment A uses the 16 raw standardized features. Experiment B uses the 8 PCA features. Both use the same stratified 80/20 split with 274 training ridings and 69 test ridings. This section includes train and test accuracy charts, loss convergence curves, runtime charts and a table of how PCA changed each result. It ends with an analysis and a summary that shows the confusion matrix and classification report for the best model.

**5. Conclusion.** This section pulls the findings together.

## Results

Test accuracy in percent for all 18 runs.

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

The best model is simulated annealing with the polynomial kernel on raw features at 72.5 percent test accuracy. Precision and recall sit between 0.71 and 0.74 for both classes.

PCA had mixed effects. It helped in four combinations and hurt in two. It made no difference in three. The biggest gain was AdaDelta with the polynomial kernel which went from 49.3 to 63.8 percent. BFGS with the RBF kernel also gained 4.3 points. The two drops were each 1.4 points.

BFGS was the steadiest optimizer and never took more than about a second. Simulated annealing was by far the slowest at about 25 seconds on the RBF and polynomial kernels. AdaDelta was the most sensitive to the choice of feature representation.

The test set has only 69 ridings so a single riding is worth about 1.4 points. Small differences between runs should be read with that in mind.

## Using the site

The Menu button in the top left opens and closes the contents sidebar. The page uses a dark theme only. The GitHub button in the top right links to this repository and the sidebar has a link to the GitHub profile. Both links work out their address from the site URL so they need no setup.

## Files

- `index.html` is the whole site with every plot embedded.
- The browser tab icon is your GitHub profile picture. It is loaded straight from GitHub so there is no icon file to add.
- `.nojekyll` tells GitHub Pages to serve the files as they are.
- `README.md` is this file.

## Publishing with GitHub Pages

Open the repo settings and choose Pages. Set the source to Deploy from a branch and pick `main` with the `/ (root)` folder. After a minute or two the site is live at `username.github.io/repo-name`.

## Updating the site

The sidebar, dark theme and GitHub links were added on top of the exported notebook. Exporting the notebook again would need those changes applied again.

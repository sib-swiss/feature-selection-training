## **1.** Fit knockoffs with Gaussian MVR knockoffs and the lasso feature statistic (default parameters) at FDR=0.2. 
## Which features are selected?

from knockpy import KnockoffFilter
from sklearn.preprocessing import StandardScaler

np.random.seed(456)
scaler_bc = StandardScaler()
X_bc_s = scaler_bc.fit_transform(X_bc.values)

kf_bc = KnockoffFilter(ksampler="gaussian", fstat="lasso")
rej_bc = kf_bc.forward(X=X_bc_s, y=y_bc.values, fdr=0.2)
selected_bc = X_bc.columns[rej_bc.astype(bool)]
print(f"Selected at FDR=0.2 ({len(selected_bc)} features):")
print(list(selected_bc))


#######
## **2.** Try different FDR levels (0.05, 0.1, 0.2). How does the number of selected features change?

for fdr in [0.05, 0.1, 0.2]:
    np.random.seed(789)
    kf_tmp = KnockoffFilter(ksampler="gaussian", fstat="lasso")
    rej_tmp = kf_tmp.forward(X=X_bc_s, y=y_bc.values, fdr=fdr)
    print(f"FDR={fdr}: {rej_tmp.sum():.0f} features selected")

#######
## **3.** Print the overlap between the knockoff-selected and the features selected by Boruta from Chapter 4.
# Features from chapter 4 (run it first to confirm)
# These are typical Boruta-confirmed features on this dataset:
boruta_typical = ["worst concave points", "worst perimeter", "worst area",
                  "mean concave points", "worst concavity", "worst radius"]

knockoff_set = set(selected_bc)
boruta_set = set(boruta_typical)
print(f"Knockoff selected: {sorted(knockoff_set)}")
print(f"Overlap with Boruta: {knockoff_set & boruta_set}")
print(f"In knockoff but not Boruta: {knockoff_set - boruta_set}")

###
## **4. (Bonus)** Knockoff selection is stochastic - 
## run the procedure 10 times and record which features are selected in each run. Which features are selected most consistently?

n_runs = 10
selection_counts = pd.Series(0, index=X_bc.columns)

for i in range(n_runs):
    np.random.seed(i * 10)
    kf_tmp = KnockoffFilter(ksampler="gaussian", fstat="lasso")
    rej_tmp = kf_tmp.forward(X=X_bc_s, y=y_bc.values, fdr=0.2)
    selection_counts += rej_tmp.astype(int)

print("Selection frequency across 10 runs:")
print(selection_counts[selection_counts > 0].sort_values(ascending=False).to_string())

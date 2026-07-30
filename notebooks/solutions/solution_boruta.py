## 2. use cross-validation to estimate the accuracy of a random forest on this dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


rf = RandomForestClassifier(n_jobs=-1)

rf_grid = GridSearchCV(rf,
                       {"ccp_alpha" : np.logspace(-6,-3,20)},
                       scoring='accuracy'
                      )
%time rf_grid.fit(X,y)

print( f"chosen hyper-parameter value: {rf_grid.best_params_}" )
print( f"cross-validated score: {rf_grid.best_score_}" )

RF_fi = rf_grid.best_estimator_.feature_importances_
print( f"{( RF_fi != 0  ).sum()}/{X.shape[1]} features with non-zero importances")

#######
## 3. use boruta to select all-relevant variables
from boruta import BorutaPy

rf = RandomForestClassifier(n_jobs=-1, ccp_alpha = rf_grid.best_params_['ccp_alpha'])
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=123, max_iter = 250 )

%time feat_selector.fit(X, y)

#######
## 4. keep only the features selected by boruta and use cross-validation to estimate the accuracy of a random forest 

## keep only the confirmed
Xf = feat_selector.transform( np.array(X) )
print("cross-validated score with confirmed features" ,cross_val_score(rf, Xf,y).mean() )


## keep only confirmed + tentative
Xf = feat_selector.transform( np.array(X) , weak = True )
print("cross-validated score with confirmed+tentative features" ,cross_val_score(rf, Xf,y).mean() )

#######
## 5. run Boruta again, but with the `perc` parameter to 80 instead of 100. How many more variables does it select?

from boruta import BorutaPy

rf = RandomForestClassifier(n_jobs=-1, ccp_alpha = rf_grid.best_params_['ccp_alpha'])
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=123, max_iter = 250 , perc=80)

%time feat_selector.fit(X, y)

## keep only the confirmed
Xf = feat_selector.transform( np.array(X) )

print("cross-validated score with 80% boruta selected features",cross_val_score(rf, Xf,y).mean())

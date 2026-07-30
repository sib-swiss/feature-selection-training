rfe = RFE( RandomForestClassifier(),
           n_features_to_select = 10,
           step = 0.1)
%time rfe.fit(X,y)

Xt = rfe.transform( X )
cross_val_score(RandomForestClassifier() , Xt, y, scoring = 'accuracy', cv=5).mean()
Xf = X.loc[:,selected_features]

ppl = Pipeline([('scalar',StandardScaler()), 
                ('model',LogisticRegression(C=np.inf))]) 
## C is the inverse strength of regularization, so C=np.inf corresponds to no regulation


cross_val_score( ppl, Xf, y, cv=5 , scoring='accuracy' ).mean()
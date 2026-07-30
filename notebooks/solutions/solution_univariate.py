## use different thresholds for variance selection, and evaluate them with cross_val_score.
## 
##     the top 50% variables genes
##     the top 0.1% variable genes
## 
## Which one yields the better cross-validated accuracy?

threshold = [50,10,0.1]
lr_ppl = Pipeline( [('scale',StandardScaler()),
                    ('model',LogisticRegression())] )


for t in threshold:
    
    VT = SelectPercentile( score_func = lambda x,_ : np.var(x , axis = 0) ,
                       percentile = t
                     )
    
    X_filtered = VT.fit_transform(X_xpr)
    
    cross_val_acc = cross_val_score(lr_ppl , X_filtered , y , cv = 5 , scoring = 'accuracy').mean()
    print(f'top {t}% : cross-validated accuracy: {cross_val_acc:.2f}')
    
#####
## select the top 10% variable genes, then use SelectKbest (use chi2 as scoring function) to get the top 500 genes. 
## Which cross-validated accuracy fo you get?


from sklearn.feature_selection import SelectKBest , chi2

VT = SelectPercentile( score_func = lambda x,_ : np.var(x , axis = 0) ,
                       percentile = 10
                     )
SKB = SelectKBest(score_func=chi2 , k = 500 )

X_filtered1 = VT.fit_transform(X_xpr)
X_filtered2 = SKB.fit_transform( X_filtered1 , y )


cross_val_acc = cross_val_score(lr_ppl , X_filtered2 , y , cv = 5 , scoring = 'accuracy').mean()

print(f'top 10% variant -> top 500 (chi2 stats): cross-validated accuracy: {cross_val_acc:.2f}')

#####
## cleaner version:


lr_ppl = Pipeline( [('variance_filter' , SelectPercentile( score_func = lambda x,_ : np.var(x , axis = 0) ,
                                                          percentile = 10
                                                         )),
                    ('chi2_filter' , SelectKBest(score_func=chi2 , k = 500 )),
                    ('scale',StandardScaler()),
                    ('model',LogisticRegression())] )


cross_val_acc = cross_val_score(lr_ppl , X_xpr , y , cv = 5 , scoring = 'accuracy').mean()
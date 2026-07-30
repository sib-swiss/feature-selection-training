
backward_selector = SequentialFeatureSelector(ppl,
                                             tol = 0.01,
                                             direction = 'backward',
                                             scoring = 'accuracy',
                                             cv = 5                                             
                                            )
%time backward_selector.fit(X,y)

print("number of selected features", backward_selector.get_support().sum() )

Xt = backward_selector.transform( X )
print("cross-validated score with selected features", cross_val_score(ppl , Xt, y, scoring = 'accuracy', cv=5).mean())

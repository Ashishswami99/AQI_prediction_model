from flaml import AutoML
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
def model(X_train,X_test,y_train,y_test):
    model=RandomForestRegressor(n_estimators=300)

    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)

    r2score=r2_score(y_pred,y_test)

    print("the r2score is ",r2score)
    

    return y_pred,r2score

from src.data_injection import load
from src.data_preprocessing import process
from src.model_build import model
def main():
    df=load()
    print(df.shape)
    X_train,X_test,y_train,y_test=process(df)
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

    y_pred,r2score=model(X_train,X_test,y_train,y_test)


if __name__ == "__main__":
    main()

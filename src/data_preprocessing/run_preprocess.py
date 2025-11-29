from .preprocess_diabetes import preprocess_diabetes_dataset

if __name__ == "__main__":
    (
        X_train_res,
        y_train_res,
        X_valid,
        y_valid,
        X_test,
        y_test,
        feature_names
    ) = preprocess_diabetes_dataset()

    print("[OK] Preprocess tamamlandı!")
    print("Train (SMOTE):", X_train_res.shape)
    print("Valid:", X_valid.shape)
    print("Test :", X_test.shape)

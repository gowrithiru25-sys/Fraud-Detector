import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib #this is what saves and loads the trained models


data = pd.read_csv("data/creditcard.csv") # load the dataset from the CSV file into a table-like structure (panda i think)


X = data.drop("Class", axis=1) # class is the column that says 1 = fraud, 0 = not fraud so basically seperating it into everything it needs to classify as fraud or not and the actual classification
y = data["Class"]                 


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) # test_size=0.2 means 20% is for testing 80% is used to train
# stratify=y makes sure both splits keep the same fraud to not-fraud ratio

# Save the test set to its own file, so the frontend can load examples
# the model has genuinely never trained on -- not the full dataset.
test_data = X_test.copy()
test_data["Class"] = y_test
test_data.to_csv("data/test_set.csv", index=False)

model = RandomForestClassifier(class_weight="balanced", random_state=42) # class_weight="balanced" tells the model "fraud is rare, pay extra attention to it"

model.fit(X_train, y_train)


predictions = model.predict(X_test) # tests the model on data it hasn't seen and print how well it did
print(classification_report(y_test, predictions, target_names=["Legit", "Fraud"]))


joblib.dump(model, "model.pkl") # saves the trained model to a file
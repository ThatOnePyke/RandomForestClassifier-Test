import pandas as pd
from sklearn.ensemble import RandomForestClassifier

traindf = pd.read_excel("train.xlsx", index_col = 0)

# extract the age column and use that to remove nan rows for age
traindf.dropna(subset = ["Age"], axis = 0, inplace = True)

# remove unneeded columns, mostly the ones with strings
# rfc can only take int/floats
X = traindf.drop(["Survived", "Name", "Ticket", "Cabin"], inplace = False, axis = 1)

# grabbing the column for survived
y = traindf["Survived"]

gender = X["Sex"].tolist()

for i in range(len(gender)):
    if (gender[i] == "male"):
        gender[i] = 0
    else:
        gender[i] = 1
        
X["Sex"] = gender

embarked = X["Embarked"].tolist()

for i in range(len(embarked)):
    if(embarked[i] == "S"):
        embarked[i] = 0
    if (embarked[i] == "Q"):
        embarked[i] = 1
    else:
        embarked[i] = 2
        
X["Embarked"] = embarked

#print(X["Embarked"].unique())

rfc = RandomForestClassifier(max_depth=2, random_state=0)

rfc.fit(X, y)

# now lets clean up the test data
testdf = pd.read_excel("test.xlsx", index_col = 0)

testdf.dropna(subset=["Age", "Fare"], axis = 0, inplace=True)

X2 = testdf.drop(["Cabin", "Ticket", "Name"], inplace = False, axis = 1)

y2 = []

gender = X2["Sex"].tolist()

for i in range(len(gender)):
    if (gender[i] == "male"):
        gender[i] = 0
    else:
        gender[i] = 1
        
X2["Sex"] = gender

embarked = X2["Embarked"].tolist()

for i in range(len(embarked)):
    if(embarked[i] == "S"):
        embarked[i] = 0
    if (embarked[i] == "Q"):
        embarked[i] = 1
    else:
        embarked[i] = 2
        
X2["Embarked"] = embarked

results = rfc.predict(X2)

print(results)
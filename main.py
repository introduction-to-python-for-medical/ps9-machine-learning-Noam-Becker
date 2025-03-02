%load_ext autoreload
%autoreload 2

# Download the data from your GitHub repository
!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O /content/parkinsons.csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/content/parkinsons.csv')
#sns.pairplot(df)
plt.show()
input_feature1 = 'MDVP:Fo(Hz)'  
input_feature2 = 'D2'  
output_feature = 'status' 
print(f"Selected input features: {input_feature1}, {input_feature2}")
print(f"Selected output feature: {output_feature}")
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
input_features = df[[input_feature1, input_feature2]]
scaled_features = scaler.fit_transform(input_features)
scaled_df = pd.DataFrame(scaled_features, columns=[input_feature1, input_feature2])
print(scaled_df.head())
from sklearn.model_selection import train_test_split
X = scaled_df
y = df[output_feature]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svm_model = SVC(kernel='linear', C=1) 
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy of SVM model: {accuracy}")

#!/usr/bin/env python
# coding: utf-8

# In[1]:


data = [
    ('<=30', 'high', 'no', 'fair', 'no'),
    ('<=30', 'high', 'no', 'excellent', 'no'),
    ('31…40', 'high', 'no', 'fair', 'yes'),
    ('>40', 'medium', 'no', 'fair', 'yes'),
    ('>40', 'low', 'yes', 'fair', 'yes'),
    ('>40', 'low', 'yes', 'excellent', 'no'),
    ('31…40', 'low', 'yes', 'excellent', 'yes'),
    ('<=30', 'medium', 'no', 'fair', 'no'),
    ('<=30', 'low', 'yes', 'fair', 'yes'),
    ('>40', 'medium', 'yes', 'fair', 'yes'),
    ('<=30', 'medium', 'yes', 'excellent', 'yes'),
    ('31…40', 'medium', 'no', 'excellent', 'yes'),
    ('31…40', 'high', 'yes', 'fair', 'yes'),
    ('>40', 'medium', 'no', 'excellent', 'no')
]

# Count instances for each class
buys_computer_yes = sum(1 for _, _, _, _, buys_computer in data if buys_computer == 'yes')
buys_computer_no = sum(1 for _, _, _, _, buys_computer in data if buys_computer == 'no')

# Total number of instances
total_instances = len(data)

# Calculate prior probabilities
prior_prob_yes = buys_computer_yes / total_instances
prior_prob_no = buys_computer_no / total_instances

# Print results
print(f'Prior probability for buys_computer="yes": {prior_prob_yes:.2f}')
print(f'Prior probability for buys_computer="no": {prior_prob_no:.2f}')


# In[11]:


import numpy as np
import pandas as pd
df=pd.read_excel("embeddingsdata (1).xlsx")
df


# In[20]:


import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import norm
import numpy as np

# Example: Creating a DataFrame
data = {
    'embed_1': np.random.normal(loc=0, scale=1, size=200),
    'embed_2': np.random.normal(loc=2, scale=1, size=200),
    'Label': np.random.choice([0, 1], size=200)
}

df=pd.read_excel("embeddingsdata (1).xlsx")
# Filter the DataFrame for labels 0 and 1
binary_dataframe = df[df['Label'].isin([0, 1])]

# Extract features and labels
X = binary_dataframe[['embed_1', 'embed_2']]
y = binary_dataframe['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Calculate class-conditional densities for the first feature in class 0
feature_index = 0
class_0_data = X_train.loc[y_train[y_train == 0].index, 'embed_1']
mu, std = norm.fit(class_0_data)
class_0_density = norm.pdf(X_train['embed_1'], mu, std)

# Check if any class-conditional density has zero values
zero_density_classes = [label for label in np.unique(y_train)
                        if np.any(norm.pdf(X_train.loc[y_train[y_train == label].index, 'embed_1'], mu, std) == 0)]

# Print the results
print(f"Class-conditional density for feature {feature_index} in class 0:\n{class_0_density}")
print(f"Classes with zero class-conditional densities for feature {feature_index}:\n{zero_density_classes}")


# In[19]:


from scipy.stats import pearsonr

# Assuming you have a DataFrame df
df=pd.read_excel("embeddingsdata (1).xlsx")
# Select the features for testing independence
features_to_test = ['embed_1', 'embed_2', 'embed_3', 'embed_4']

# Calculate the Pearson correlation coefficient and p-value for each pair of features
for i in range(len(features_to_test)):
    for j in range(i + 1, len(features_to_test)):
        feature1 = features_to_test[i]
        feature2 = features_to_test[j]
        
        correlation_coefficient, p_value = pearsonr(df[feature1], df[feature2])
        
        print(f"Correlation between {feature1} and {feature2}: {correlation_coefficient}")
        print(f"P-value: {p_value}")
        print("")


# In[22]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Assuming you have a DataFrame 'df' containing your data
# If not, you can create one using the provided data

# Encode categorical variables
le_age = LabelEncoder()
le_income = LabelEncoder()
le_student = LabelEncoder()
le_credit_rating = LabelEncoder()
le_buys_computer = LabelEncoder()

df['Age'] = le_age.fit_transform(df['Age'])
df['Income'] = le_income.fit_transform(df['Income'])
df['Student'] = le_student.fit_transform(df['Student'])
df['Credit Rating'] = le_credit_rating.fit_transform(df['Credit Rating'])
df['Buys Computer'] = le_buys_computer.fit_transform(df['Buys Computer'])

# Define features (X) and labels (y)
X = df.drop('Buys Computer', axis=1)
y = df['Buys Computer']

# Split the data into training and testing sets
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Gaussian Naive Bayes model
model = GaussianNB()
model.fit(Tr_X, Tr_y)

# Now, you can use the trained model to make predictions on new data
predictions = model.predict(Te_X)

# Evaluate the model, if needed
accuracy = model.score(Te_X, Te_y)
print(f"Accuracy: {accuracy}")


# In[21]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Assuming you have a DataFrame df with your dataset
# df = ...

# Specify the features (X) and labels (y)
features = ['embed_1', 'embed_2', 'embed_3', 'embed_4']
target_variable = 'Label'

X = df[features]
y = df[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build the Naïve-Bayes (NB) classifier
model = GaussianNB()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Print the results
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report_str)


# In[ ]:





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
This python file is responsible for cleaning and processing
the data to be able to feed it to the Neural Network

"""

def data_process(data_path):

    # Load the CCD dataset path
    df = pd.read_excel(data_path, header=[0, 1])  
    # Specify int that headers are present in the first two rows to be able to remove them
    # Remove any leading/trailing whitespace from column names
    df.columns = df.columns.map(lambda x: (x[0].strip(), x[1].strip()))
    
    # Handling missing values, dropna just drops the rows with missing (NaN) values
    df.dropna(inplace=True)
    
    
    # Encoding categorical variables
    df = pd.get_dummies(df, columns=[('X2', 'SEX'), ('X3', 'EDUCATION'), ('X4', 'MARRIAGE')])
    
    
    # Splitting the data into features (X) and target variable (y)
    X = df.drop(columns=[('Y', 'default payment next month')])  # Use a tuple for multi-level columns
    # data split : split the data, X is data without Y column and Y only with that one column *One Dimensional*
    y = df[('Y', 'default payment next month')]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert column names to strings
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    # Scaling the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train = X_train_scaled
    X_test = X_test_scaled

    #return our data's training and test sets as well as data's validation, Y
    return X_train, X_test, y_train, y_test
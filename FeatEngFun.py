def titanic_feature_eng(data, DATASET, drop_all, cabin_pca, cabin_group_encoding):

    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from scipy.spatial import ConvexHull

    # Defining the encoders:
    label_encoder = LabelEncoder()
    scaler = StandardScaler()

    # Dropping the 'Ticket', 'Name', 'Embarked', 'Parch' and 'SibSp' column.
    if drop_all:
        data = data.drop(['Ticket', 'Name', 'Embarked', 'Parch', 'SibSp'], axis = 1)


    # Binary Encoding of the 'Sex' Column'
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})


    # Replacing the missing values in the 'Age' column by the mean of 
    # the column.
    data['Age'] = data['Age'].fillna(data['Age'].mean())

    # Keeping only the first letter of the first Cabin in which the pas-
    # senger / family member is located in. This choice is justified by
    # the layout of cabins on the Titanic, with cabins in alphabetical 
    # order representing decks from the top to the bottom of the ship.
   
    data['Cabin'] = data['Cabin'].apply(lambda x: x.split(' ')[0][0] if pd.notna(x) else np.nan)


    # Label encoding of the cabins.
    original_cabins = data['Cabin']
    #data['Cabin'] = data['Cabin'].apply(lambda x: label_encoder.fit_transform([x])[0] if pd.notna(x) else np.nan)


    # Standardising the 'Fare' and 'Age' columns.
    standardized_columns = ['Age', 'Fare']
    """data[standardized_columns] = scaler.fit_transform(data[standardized_columns])  """
    
    if 'Survived' in data.columns:

        print('The dataset given is the training set.')


    ### --- PERFORMING CABIN CLUSTERING  --- ###

        if cabin_pca:

            # Performing a PCA on the data set, reducing the number of 
            # features to 3. Note that this is performed on the passengers
            # for which a cabin is provided. This clustering operation
            # is then used to identify the segmentation between the dif-
            # ferent cabins, whether they should be grouped or not, and 
            # to ease the cabin imputing process thereafter.

            data_clean_cabins = data.dropna(subset = ['Cabin'], axis = 0)
            data_clean_cabins['Cabin'] = label_encoder.fit_transform(data_clean_cabins['Cabin'])
            data_clean_cabins[standardized_columns] = scaler.fit_transform(data_clean_cabins[standardized_columns]) 
            #print(data_clean_cabins)

            pca = PCA(n_components = 3)
            X_pca = pca.fit_transform(data_clean_cabins)


            # Initialising the KMeans algorithm and using the elbow method to 
            # determine the optimal number of clusters.

            inertia = []

            for i in range(1,15):
                KM = KMeans(init = 'k-means++', n_clusters = i, random_state = 42)
                train_results = KM.fit(X_pca)
                inertia.append(KM.inertia_)


            # Plotting the cluster inertia vs number of clusters.
            plt.figure()
            sns.lineplot(x = list(range(1,15)), y = inertia, marker = 'o')


            # After having decided upon the number of clusters to retain
            # a second KMeans clustering is performed.

            KM = KMeans(init = 'k-means++', n_clusters = 2, random_state = 42)
            train_results = KM.fit(X_pca)

            data_clean_cabins['cluster'] = KM.labels_
            data_clean_cabins['CabinLetter'] = original_cabins

            # Creating a crosstab
            corrtable = pd.crosstab(data_clean_cabins['cluster'], data_clean_cabins['CabinLetter'], normalize=True)

            # Now, normalize the crosstab per Cabin Letter group
            corrtable_normalized = corrtable.div(corrtable.sum(axis=0), axis=1)
            print(corrtable_normalized) 

        # When 2 clusters are created it can be seen that the groups
        # can be created, one with cabins 'A-B-C' and the other with
        # the rest of the cabins, making it easier to impute the 
        # missing cabins thereafter.
        

        ### --- Encoding the Cabin Groups --- ###


    if cabin_group_encoding:
        
        # Defining a function to encode the cabins into 0 and 1s.
        data['CabinLetter'] = original_cabins
        def cabin_group_encoding_fun(cabin_letter):

            # Defining the Cabin Groups to encode into 0's and 1's.
            cabin_groups = [['A', 'B', 'C'], ['D', 'E', 'F', 'G', 'T']]
            if cabin_letter in cabin_groups[0]:
                return 0
            elif cabin_letter in cabin_groups[1]:
                return 1
            else:
                return np.nan

        data['Cabin'] = data['CabinLetter'].apply(cabin_group_encoding_fun)

    ### --- DEFINING A CABIN IMPUTING METHOD --- ### MAYBE DEFINE A FUNCTION FOR THIS ALONE?

    # 1. Based on the closest average fare Price
    # 2. Using a classification model

    
            
    return data, original_cabins



# Defining a function to determine the cabin of a passenger based on the
# fare price paid and the closest average cabin fare price.
def closest_cabin_avg_fare_price(row, cabin_avg_fare):
    import numpy as np

    distance = {}
    for cabin, fare in cabin_avg_fare.items():
        distance[cabin] = np.abs(fare - row['Fare'])/fare

    return min(distance, key = distance.get)


# Defining a function imputing the cabins for the passengers missing one
# by using the 'closest_cabin_avg_fare_price' function above.
def cabin_imputing_fun(dataframe):
    import numpy as np
    import pandas as pd

    # Calculating the average fare price per cabin/group of cabin.
    cabin_avg_fare = dataframe.loc[pd.notna(dataframe['Cabin'])].groupby('Cabin')['Fare'].mean()
    #unique_cabins = sorted(df['cabins_stripped'].unique())
    print('Average prices for cabins depending on the cabin floor are the following: \n',cabin_avg_fare)


    #cabin_na = dataframe.loc[dataframe['Cabin'].isna(), 'Fare']

    # Creating an array containing the cabins imputed for the passengers
    # who miss one, based on the method developed above.
    imputed_cabins = dataframe.loc[dataframe['Cabin'].isna()].apply(lambda row: closest_cabin_avg_fare_price(row, cabin_avg_fare), axis = 1)

    # Replacing the NaN values of cabins by the imputed cabins.
    dataframe.loc[dataframe['Cabin'].isna(), 'Cabin'] = imputed_cabins
    #print(dataframe[['Fare', 'Cabin']].head(10))

    return dataframe
# This file contains the general function used to perform a gridsearch 
# on chosen models. At this step of the project it was chosen to define
# the models to be gridsearched by hand in the 'models_dict' dictionary.
# It is planned to implement the possibility for the user to choose the
# models to be gridsearched in the future using a UI.

def gridsearch_fun(X_train, X_test, y_train, y_test):

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report
    import xgboost as xgb
    from xgboost import XGBClassifier
    from sklearn.linear_model import LogisticRegression
    import pickle
    import numpy as np
    import pandas as pd

    # Defining the list of models to be tested on the trainig dataset.
    models_dict = {1: 'Logistic Regression', 2: 'Random Forest', 3: 'Support Vector Machine', 4: 'Support Vector Machine 2', 5: 'XGBoost'}

    # Choosing which model to use, 1-5 if all models are to be gridsearched
    model_numbers = [1,2,3,4,5]

    # Initialising an array containing the names of the model to be used 
    # during the gridsearch.
    enable_grid_search = []

    for element in model_numbers:
        enable_grid_search.append(models_dict[element])

    with open('enable_grid_search', 'wb') as f:
        pickle.dump(enable_grid_search, f)

    # Initialising models and their respective hyperparameters to be kept
    # fixed throughout the grid search.
    models = [
        ('Logistic Regression', LogisticRegression(solver='liblinear', max_iter=10000, class_weight='balanced', random_state = 42)),
        ('Random Forest', RandomForestClassifier(n_estimators = 100, class_weight='balanced', criterion='entropy', max_depth=5, random_state = 42)),
        ('Support Vector Machine', SVC(probability=True, class_weight='balanced', random_state = 42)),
        ('Support Vector Machine 2', SVC(probability=True, class_weight='balanced', random_state = 42)),
        ('XGBoost', XGBClassifier(objective = 'binary:logistic', eta = 0.4, max_depth = 1, random_state = 42))
    ]



    # Defining the hyperparameters to test out for each model for the gridsearch

    # Define parameter grids for each model
    parameter_grid_full = {
        'Logistic Regression': {
            'classifier__C': np.linspace(0.1,10,10),
            'classifier__penalty': ['l1', 'l2']
        },
        'Random Forest': {
            'classifier__n_estimators': np.linspace(110,140,31).astype(int),
            #'classifier__max_depth': list(range(5))
        },
        'Support Vector Machine': {
            'classifier__C': np.linspace(0.1, 1, 10),
            'classifier__kernel': ['linear', 'rbf', 'poly', 'sigmoid']
        },
        'Support Vector Machine 2': {
            'classifier__C': np.linspace(0.1, 1, 10),
            'classifier__kernel': ['poly'],
            'classifier__degree': [1, 2, 3]
        },
        'XGBoost': {
            #'classifier__eta': np.linspace(0.3, 0.6, 5),
            #'classifier__max_depth': np.linspace(1, 11, 6).astype(int),
            'classifier__n_estimators': np.linspace(190,210,21).astype(int),
            'classifier__gamma': np.linspace(0.01, 0.5, 20)
        }
    }

    # Initialising empty lists to store the results of the best models

    gs_results = [] # To store the results of the gridsearch
    perf_reports = [] # To store the performance reports of each model and print them afterwards.
    updated_models = {} # To store the best versions of each model with their updated hyperparameters.
    
    # Defining the labels to use in the classification reports.
    target_names = ['Not Survived', 'Survived']

    # Individually fitting the different models to the training set in an iterative manner,
    # before predicting on the test set and outputting the classification report for each model.

    for n, m in models:

        # Filtering which model to perform the grid search on
        if n not in enable_grid_search:
            continue
        #print(f"{n} \n")

        pipeline = Pipeline([
            ('classifier', m)
        ])

        parameter_grid = parameter_grid_full[n] 
        grid_search = GridSearchCV(pipeline, 
                                parameter_grid, 
                                cv = 2, 
                                scoring='accuracy', 
                                refit = 'accuracy',
                                verbose = True)
        
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        predictions = best_model.predict(X_test)

        updated_models[n] = best_model.named_steps['classifier']
        
        print(f"\n\nThe best version of the {n} model is :\n\n {best_model.named_steps['classifier']} \n\n")
    

        with open('updated_models', 'wb') as f:
            pickle.dump(updated_models, f)

        ## Storing the parameters and the score of the best instance of each of the models tested.
        gs_results.append({
            'model': n,
            'best_parameters': grid_search.best_params_,
            'best_score': grid_search.best_score_,
        })

        perf_reports.append({
            'n': ('classification_report', classification_report(y_test, predictions, target_names=target_names))
        })

        results = pd.DataFrame(grid_search.cv_results_)
        results = results.sort_values(by = 'mean_test_score', ascending=False)


        ## Defining the name of the file in which the results of the grid search will be output.

        n_clean = n.replace(' ', '_')
        file_name = f'{n_clean}_GS_results.csv'

        results.to_csv(file_name, index = False)


    # Returning the final output from the gridsearch

    return enable_grid_search
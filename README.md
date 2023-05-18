# Spaceship Titanic
In the year 2912, the Spaceship Titanic, carrying thousands of passengers on a voyage to habitable exoplanets, collided with a spacetime anomaly. In this project I'm analyzing the recovered records from the spaceship's damaged computer system and I'm predicting which passengers were transported to an alternate dimension. See how it went, what results I got and what I learned along the way.

![C66ez28](https://github.com/CodecoolGlobal/spaceship-titanic-python-IzabelMatusiewicz/assets/101067795/e4f359e2-b842-4e20-b659-775391534055)

## About the project
Based on the Spaceship Titanic dataset, I utilized XGBoost, LightGBM, and CatBoost models, and incorporated Optuna for hyperparameter optimization. Additionally, I explored the PyTorch library by implementing linear regression and employed k-fold cross-validation techniques

## Methodology
### Preprocessing
1. Transforming boolean values to float
2. Dropping columns with uniqe values 
3. Categorical data One-Hot Encoding
4. Missing values from numerical column imputer with IterativeImputer()
5. Numerical columns standarization with StandardScaler()

### Boosting methods implementation
#### LightGBM
Optimum parameters founded with Optuna:
| Parameter |Value |
|----------|---------|
| colsample_bytree | 0.6635086134898114 |
| learning_rate | 0.5117257983113679 |
| max_depth | 2 |
| min_child_samples | 10 |
| min_child_weight | 0.00022048430077266904 |
| n_estimators | 65 |
| num_leaves | 14 |
| reg_alpha | 0.060205308471712515 |
| reg_lambda | 0.032379685342864915 |
| subsample | 0.5648029088154599 |

### XGBoost
Optimum parameters founded with Optuna:
| Parameter           | Value                  |
|---------------------|------------------------|
| n_estimators        | 105                    |
| max_depth           | 10                     |
| learning_rate       | 0.08033125836409204    |
| subsample           | 0.7345205255792372     |
| colsample_bytree    | 0.696155441528216      |
| gamma               | 4.652541314742388e-06  |
| min_child_weight    | 1                      |
| reg_alpha           | 3.273763178535857e-06  |
| reg_lambda          | 0.00022652047351151444 |

### CatBoost
Optimum parameters founded with Optuna:
| Parameter             | Value                  |
|-----------------------|------------------------|
| iterations            | 392                    |
| depth                 | 9                      |
| learning_rate         | 0.22460498554077832    |
| random_strength       | 20                     |
| bagging_temperature   | 0.9061464086574511     |
| border_count          | 42                     |


### Boosting models summary
The obtained results from the three models, LightGBM, XGBoost, and CatBoost, exhibit similar performance with slight variations. Considering the above results, LightGBM appears to have the highest overall performance across the evaluated metrics. However, it is worth noting that the differences between the models' performances are relatively small, indicating that all three models are capable of accurately predicting the anomalies related to the Spaceship Titanic dataset.

## Pytorch implementation
to be added

## Results summary
### LightGBM
| Metric         | Result            |
|-----------------|---------------------|
| Accuracy score  | 0.799079754601227   |
| Recall score    | 0.8454404945904173  |
| F1-score        | 0.8067846607669615  |
| ROC AUC         | 0.8785084654852393  |

![download (23)](https://github.com/CodecoolGlobal/spaceship-titanic-python-IzabelMatusiewicz/assets/101067795/b6d4fd1b-8452-41ff-9b99-00831601ea8d)

### XGBoost
| Metric          | Value                    |
|-----------------|--------------------------|
| Accuracy Score  | 0.7898773006134969       |
| Recall Score    | 0.8299845440494591       |
| F1-score        | 0.7967359050445103       |
| ROC AUC         | 0.8715379964665391       |

![download (24)](https://github.com/CodecoolGlobal/spaceship-titanic-python-IzabelMatusiewicz/assets/101067795/6a2527d5-0f8c-4c5f-8453-0ded62018e95)

### CatBoost

| Metric         | Value                |
|----------------|----------------------|
| Accuracy score | 0.799079754601227    |
| Recall score   | 0.8361669242658424   |
| F1-score       | 0.8050595238095238   |
| ROC AUC        | 0.8783390852053382   |

![download (25)](https://github.com/CodecoolGlobal/spaceship-titanic-python-IzabelMatusiewicz/assets/101067795/f668cda5-3bdb-46c3-ad0c-b791002acb03)


## Summary

XGBoost, LightGBM, and CatBoost are widely used libraries for gradient boosting in predictive modeling. Each library has its unique strengths and differences. XGBoost is a versatile and widely adopted approach, offering scalability, efficient handling of large datasets, and various advanced features such as regularization and handling missing data. It provides reliable predictive performance but may be relatively slower when dealing with extensive datasets.

On the other hand, LightGBM is known for its exceptional performance and speed. It is optimized for efficient memory usage and parallel processing. LightGBM employs a leaf-wise tree growth strategy, which differs from the traditional approach used by XGBoost. This technique can yield improved results, particularly when working with large datasets.

CatBoost stands out for its specialized handling of categorical features without requiring explicit encoding. It also includes built-in support for handling missing data and automatic hyperparameter optimization. CatBoost employs techniques like feature encoding and recursive learning of the tree structure, making it particularly useful when dealing with datasets containing a substantial number of categorical features.

In this case, the results from XGBoost, LightGBM, and CatBoost were nearly identical. It suggests that the dataset was well-suited for these models, and the parameters were appropriately optimized. However, it's important to remember that results can vary depending on the specific characteristics of the dataset and the nature of the problem at hand. Therefore, it's important to explore and experiment with different models, parameters, and techniques to find the best approach for your particular scenario.

## Contact
[LinkedIn profil](https://www.linkedin.com/in/izabela-matusiewicz/)

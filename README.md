# Disaster Response Project
### Udacity Data Science Nanodegree Project - build a model for an API that classifies disaster messages

In this project we will analyze the disaster data from [Figure Eight] (https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.

### Extract, Transform, and Load process (ETL) pipeline module
In order to process data and execute the ETL pipeline, the Python script **data/process_data.py** must be executed with Python. This script has the following mandatory arguments:
   1. Disaster messages data in CSV format
   2. Categories messages data in CSV format
   3. The name of the database where the processed data will be saved. The messages will be saved into a table called **tabMessages**

> **python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db**

[exampleETL]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Execute_ETL_Pipeline.png "Example ETL execution"
![alt text][exampleETL]

### The machine learning (ML) pipeline module
In order to build and train the models and execute the ML pipeline, the Python script **models/train_classifier.py** must be executed with Python. This script has the following mandatory arguments:
   1. The name of the database where the processed data has been saved in the previous step.
   2. The name of the pickle file where the models will be saved.
   
> **python models/train_classifier.py data/DisasterResponse.db models/savedModels.pkl**

[exampleML]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Execute_ML_Pipeline.png "Example ML execution"
![alt text][exampleML]

The ML pipeline module will do the following main steps:
   - Load data from the SQLite database and preprocess all messages (tokenize them)
   - Build a machine learning pipeline
   - Train one model for every category that has to be predicted. A GridSearchCV for automatic tunning will be performed for every model to be trained. The GridSearchCV will use a recall scoring method with a macro average, in order to ensure that for the different categories the class 1 will have the most important weight.
   
### Flask Web App
In order to build and train the models and execute the ML pipeline, the Python script **runApp.py** must be executed with Python. 
   
> **python runApp.py**

[exampleApp]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Execute_Flask_Application.png "Example Flask Application execution"
![alt text][exampleApp]

The application will accept the filling of a new message and will output the linked categories.

[message]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Disaster_Response_Project_01.png "Disaster Response Project Message"
![alt text][message]
   
### Data exploration
The dataset contains feature << genre >> that has the following possible values:
    - direct
    - news
    - social

This feature is one of the most balanced one in the dataset. It has the following distribution:

[genreDistribution01]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Disaster_Response_Project_02_01.png "Disaster Response Project Genre distribution 01"
![alt text][genreDistribution01]

[genreDistribution02]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Disaster_Response_Project_02_02.png "Disaster Response Project Genre distribution 02"
![alt text][genreDistribution02]

[genreDistribution03]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Disaster_Response_Project_02_03.png "Disaster Response Project Genre distribution 03"
![alt text][genreDistribution03]

[genreDistribution04]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Disaster_Response_Project_02_04.png "Disaster Response Project Genre distribution 04"
![alt text][genreDistribution04]

The other categories that have to be predicted have are quite unbalanced. Most of them have class 0 in most of the cases.

[categoriesDistribution01]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Disaster_Response_Project_03.png "Disaster Response Project Categories distribution 01"
![alt text][categoriesDistribution01]

[categoriesDistribution02]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Disaster_Response_Project_04.png "Disaster Response Project Categories distribution 02"
![alt text][categoriesDistribution02]

[categoriesDistribution03]: https://github.com/lisaro82/___Disaster-Response/blob/master/screenShots/Disaster_Response_Project_05.png "Disaster Response Project Categories distribution 03"
![alt text][categoriesDistribution03]

For the prediction of these categories, we are using for the parameter <<class_weight>> for the model either the value balanced, or a custom calculated class weight.

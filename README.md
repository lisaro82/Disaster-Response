# ___Disaster-Response
Udacity Data Science Nanodegree Project - build a model for an API that classifies disaster messages

In order to process data and execute the **ETL (Extract, Transform, and Load process) pipeline**, the Python script **data/process_data.py** must be executed with Python. This script has the following mandatory arguments:
   1. Disaster messages data in CSV format
   2. Categories messages data in CSV format
   3. The name of the database where the processed data will be saved. The messages will be saved into a table called **tabMessages**

#### python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

[exampleETL]: https://github.com/lisaro82/___Disaster-Response/blob/master/examples/Execute_ETL_Pipeline.png "Example ETL execution"
![alt text][exampleETL]

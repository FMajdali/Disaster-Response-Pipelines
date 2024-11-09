# Disaster-Response-Pipelines
This respiratory is my submission for the Disaster Response Pipelines project for the Data Scientist Nanodegree by Udacity, the project aims to classify disaster messages by building a Machine learning model on historical data.

# Files
1. disaster_categories.csv: this file contains the categories of each message ID, which the ML model will train on, all messages are categorized by a long text, and each category is followed by a dash then 1 or 0 indicating if the message belongs to this category or not
 
2. disaster_messages.csv: This file contains the messages, the text of the message in it is the original language, and the genre of the message

3. process_data.py: this script takes both files above as input, transforms the disaster_categories.csv into columns format, where each category has one column representing it, then it merges that with disaster_messages.csv into a pandas data frame, this dataframe is then saved as a sqlite database by the name of DisasterResponse.db as an output of this script.

4. DisasterResponse.db: This is my output from process_data.py, I have added it for reference

5. train_classifier.py: this script reads the database produced by process_data.py, which in our case is DisasterResponse.db, and then it builds a sklearn pipeline, this pipeline engineers two main features:
 * Build a term-document matrix using a custom tokenizer, then apply TF-idf algorithm to it as a feature
 * Make three features that represent: the message length, the number of words in the message, and if the specific words ("water", "food", "earthquake") are in the message

Then the pipeline feeds these features into a multi-output random forest, after that, there is an evaluation function that evaluates the performance of the model and print metrics.
The script also saves the ML model into a pickle file, which in our case is train_classifier.py

6. classifier.pkl: This is my output from train_classifier.py, I have added it for reference

7. run.py: This is a web app that takes both DisasterResponse.db and classifier.pkl as input, then it has a UI which allows the user to enter a message and it will provide the predicted classification 

# Running the App
1. Run the following commands in the files root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database you need to run process_data.py while passing the paths of disaster_messages.csv, disaster_categories.csv, and the path for the output database,
      for example:
        `python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves a pkl model you need to run train_classifier.py while passing the path of the cleaned messages database and the name for the output model,
      For example
        `python train_classifier.py DisasterResponse.db classifier.pkl`
      
2. Run your web app: `python run.py`, the path having the run.py script should also contain the folder templates so app could work proprly 

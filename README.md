# Disaster-Response-Pipelines
This respiratory is my submission for the Disaster Response Pipelines project for the Data Scientist Nanodegree by Udacity, the project aims to classify disaster messages by building a Machine learning model on historical data.

# Files
1. disaster_categories.csv: this file contains the categories of each message id, which the ML model will train on, all messages are catogrized by a long text, each category is followed by a dash then 1 or 0 indicating if the message is belonging to this category or not
 
2. disaster_messages.csv: This file contains the messages, the text of the message in it is original language, and the genre of the message

3. process_data.py: this script takes both files above as an input, it transforms the disaster_categories.csv into columns format, where each category has one column representing it, then it merges that with disaster_messages.csv into pandas dataframe, this dataframe is then saved as a sqlite database by the name of DisasterResponse.db as an output of this script.

4. DisasterResponse.db: This is my output from process_data.py, I have added it for reference

5. train_classifier.py

6. classifier.pkl: This is my output from train_classifier.py, I have added it for reference

7. run.py


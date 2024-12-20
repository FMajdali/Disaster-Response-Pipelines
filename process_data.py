import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine 

#This function loads messages and categories from CSV files and merge them into one pandas dataframe as an output
def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv( messages_filepath ,dtype = str)
    categories = pd.read_csv(categories_filepath, dtype= str)
    return pd.merge(messages,categories, on = "id")


#This function cleans the dataframe catgories column by sperating each catgory into one column
def clean_data(df):
    categories = df.categories.str.split(";", expand= True)
    row = categories.iloc[[0]]
    category_colnames = [row[x].iloc[0][:-2] for x in range(row.shape[1])]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1] 
        # convert column from string to numeric
        categories[column] = categories[column].apply(int)
    df.drop(columns = ["categories"], inplace = True)
    #removing child_alone column as it include no value
    categories.drop(columns=["child_alone"], inplace= True)
    # making a new column "not_relevent", if all catgories are 0, then this column is 1,
    #because this is causing issue for some 
    # ML algorithmes in the next part
    lst = []
    for row in range(categories.shape[0]):
        if categories.iloc[row].sum() == 0:
            lst.append(1)
        else:
            lst.append(0)

    categories["not_relevent"] = lst
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    # drop duplicates
    df.drop_duplicates(subset=["message"],inplace= True)
    
    return df.copy()

#save the dataframe into an SQL database
def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

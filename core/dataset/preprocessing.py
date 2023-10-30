import sys 
import os

import re
import csv
import glob

from collections import OrderedDict
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import train_test_split

import pdb

class preprocessing:
    def __init__(self, id2label, test_size: int=0.2, val_size: int=0.125, min_sentence_len: int=2) -> None:
        self.id2label = id2label
        self.test_size = test_size
        self.val_size = val_size
        self.min_sentence_len = min_sentence_len



    def filter_special_characters(self, tweet):
        # Convert to lowercase
        tweet = tweet.lower()
        
        # Remove URLs
        tweet = re.sub(r'http\S+', '', tweet)
        
        # Remove mentions
        tweet = re.sub(r'@\w+', '', tweet)
        
        # Remove the hashtag symbol only
        tweet = tweet.replace('#', '')
        
        # Remove special characters, numbers, and punctuations except for the apostrophe and slash
        tweet = re.sub(r"[^a-z\s'/]", '', tweet)
        
        return tweet

    def process_and_save_to_csv(self, raw_path, output_dir, label):
        # Read the input CSV file
        with open(raw_path, 'r') as infile:
            reader = csv.reader(infile)
            # Skip header if any; if there's no header, comment the next line out
            next(reader)
            tweets = [row[0] for row in reader]

        """ 
        Prefiltering
        """
        processed_tweets = []
        for tweet in tweets:
            # Special Char
            filtered_tweet = self.filter_special_characters(tweet)
            # Tokenize: Split the tweet into words
            tokenized_tweet = filtered_tweet.split()
            # Further split words on the slash and flatten the list
            tokenized_tweet = [item for sublist in [word.split('/') for word in tokenized_tweet] for item in sublist]
            if len(tokenized_tweet) >= self.min_sentence_len:
                joined_tweets = "\t".join(tokenized_tweet)
                processed_tweets.append(joined_tweets)

        # Split into Train, Test, (Val)
        emoji_label = [label] * len(processed_tweets) # BUG: writerow() only accepts list as input, so we need to wrap the interger with list
        x_train, x_test, y_train, y_test = train_test_split(processed_tweets, emoji_label, test_size=self.test_size) # TODO: Pass random_state to control the random seed that determines the state
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_size) 
        output = [zip(y_train, x_train), zip(y_val, x_val), zip(y_test, x_test)]
        
        # Save to CSV
        output_paths = []
        output_filenames = ["train", "val", "test"]
        for output_filename in output_filenames:
            output_paths.append(output_dir + output_filename + ".csv")
        
        for output_data, output_path in zip(output, output_paths):
            with open(output_path, 'a', newline='') as out_file:
                writer = csv.writer(out_file)
                # writer.writerow(["Specified Unicode", "Processed Tweet"])  # Writing the headers
                for data in output_data:
                    writer.writerow(data)

    def process_all_csvs_in_directory(self):
        output_dir = "core/dataset/data/processed/"
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for label, raw_path in self.id2label.items():
            raw_path = "core/dataset/data/raw/" + raw_path + ".csv"
            self.process_and_save_to_csv(raw_path, output_dir, label)
    

        # # Loop over all CSV files in the input directory
        # for csv_file in glob.glob(os.path.join(input_directory, '*.csv')):
        #     # Derive the output filename based on the input filename
        #     base_name = os.path.basename(csv_file)
        #     output_filename = os.path.join(output_directory, base_name)
            
        #     # Process the CSV file
        #     # print(csv_file)
        #     process_and_save_to_csv(csv_file, output_filename, emoji_unicode_list[i])
        #     i = i + 1
        
# id2label = {0: "cooking", 
#              1: "sun", 
#              2:"clown_face", 
#              3: "ghost", 
#              4: "skull", 
#              5: "folded_hands", 
#              6: "red_heart", 
#              7: "fire", 
#              8: "hot_face", 
#              9: "sparkles"}
id2label = {0: "enraged_face", 
             1: "face_holding_back_tears", 
             2:"face_savoring_food", 
             3: "face_with_tears_of_joy", 
             4: "fearful_face", 
             5: "hot_face", 
             6: "sun", 
             7: "loudly_crying_face", 
             8: "smiling_face_with_sunglasses", 
             9: "thinking_face"}
preprop = preprocessing(id2label)
preprop.process_all_csvs_in_directory()

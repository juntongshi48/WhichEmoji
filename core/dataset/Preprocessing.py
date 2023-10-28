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
    def __init__(self, raw_paths, test_size: int=0.2, min_sentence_len: int=2) -> None:
        self.raw_paths = raw_paths
        self.test_size = test_size
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
        output = train_test_split(processed_tweets, emoji_label, test_size=self.test_size) # TODO: Pass random_state to control the random seed that determines the state
        output = [zip(output[2], output[0]), zip(output[3], output[1])]
        
        # Save to CSV
        output_paths = []
        output_filenames = ["train", "test"]
        for output_filename in output_filenames:
            output_paths.append(output_dir + output_filename + ".csv")
        for output_data, output_path in zip(output, output_paths):
            with open(output_path, 'a', newline='') as out_file:
                writer = csv.writer(out_file)
                # writer.writerow(["Specified Unicode", "Processed Tweet"])  # Writing the headers
                for data in output_data:
                    writer.writerow(data)

    def process_all_csvs_in_directory(self):
        output_dir = "data/processed/"
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for label, raw_path in self.raw_paths.items():
            raw_path = "data/raw/" + raw_path + ".csv"
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
        
print("hi")
raw_paths = {0: "cooking", 1: "sun", 2:"clown_face"}
preprop = preprocessing(raw_paths)
preprop.process_all_csvs_in_directory()
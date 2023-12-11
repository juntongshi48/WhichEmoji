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
    def __init__(self, id2label, test_size: int=0.2, val_size: int=0.125, min_sentence_len: int=2, multi_class_label=False) -> None:
        self.id2label = id2label
        self.test_size = test_size
        self.val_size = val_size
        self.min_sentence_len = min_sentence_len
        self.multi_class_label=multi_class_label
        self.target_emojis = [
            '\U0001F621',  # enraged_face
            '\U0001F979',  # face_holding_back_tears
            '\U0001F60B',  # face_savoring_food
            '\U0001F602',  # face_with_tears_of_joy
            '\U0001F628',  # fearful_face
            '\U0001F975',  # hot_face
            '\U0001F607',  # smiling_face_with_halo
            '\U0001F62D',  # loudly_crying_face
            '\U0001F60E',  # smiling_face_with_sunglasses
            '\U0001F914',  # thinking_face
        ]



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
    
    def remove_emojis(self, text):
        if not isinstance(text, str):  # Check if text is not a string
            return text  # Return the original data if not a string
        emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F700-\U0001F77F"  # alchemical symbols
                            u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                            u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                            u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                            u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                            u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                            u"\U00002702-\U000027B0"  # Dingbats
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def label_tweet(self, tweet):
        return [1 if emoji in tweet else 0 for emoji in self.target_emojis]

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
        labels = []
        for tweet in tweets:
            # If need multiclass labels, get a list of multiclass labels in one-hot
            if self.multi_class_label:
                label_list = self.label_tweet(tweet)
                # Drop the tweet if no emoji is found
                if all([label == 0 for label in label_list]):
                    continue
            else:
                label_list = [label]
            # Filter Emojis
            filtered_tweet = self.remove_emojis(tweet)
            # Filter Special Char
            filtered_tweet = self.filter_special_characters(filtered_tweet)
            # Tokenize: Split the tweet into words
            tokenized_tweet = filtered_tweet.split()
            # Further split words on the slash and flatten the list
            tokenized_tweet = [item for sublist in [word.split('/') for word in tokenized_tweet] for item in sublist]
            if len(tokenized_tweet) >= self.min_sentence_len:
                joined_tweets = "\t".join(tokenized_tweet)
                processed_tweets.append(joined_tweets)
                labels.append(label_list)

        # Split into Train, Test, (Val)
        x_train, x_test, y_train, y_test = train_test_split(processed_tweets, labels, test_size=self.test_size)
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
        # output_dir = "core/dataset/data/multilabel/" # output dir for multilabel
        # If output dir exists and data has been processed, delete the processed data.
        # Else, create a output dir
        if os.path.exists(output_dir):
            filelist = os.listdir(output_dir)
            for filename in filelist:
                os.remove(os.path.join(output_dir, filename))
        else:
            os.makedirs(output_dir)

        for label, raw_path in self.id2label.items():
            raw_path = "core/dataset/data/raw/" + raw_path + ".csv"
            self.process_and_save_to_csv(raw_path, output_dir, label) # set to false to disable multilabel
    

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
             6: "smiling_face_with_halo", 
             7: "loudly_crying_face", 
             8: "smiling_face_with_sunglasses", 
             9: "thinking_face"}

preprop = preprocessing(id2label, min_sentence_len=10, multi_class_label=True)
preprop.process_all_csvs_in_directory()


emoji_unicode_list = [
    '\U0001F60B',  # face_savoring_food
    '\U0001F95A',  # egg
    '\U0001F628',  # fearful_face
    '\U00002600',  # sun
    '\U0001F440',  # eyes
    '\U0001F449',  # backhand_index_pointing_right
    '\U0001F970',  # smiling_face_with_hearts
    '\U00002764',  # red_heart
    '\U0001F923',  # rolling_on_the_floor_laughing
    '\U00002714',  # check_mark
    '\U0001F97A',  # face_holding_back_tears
    '\U0001F4A9',  # pile_of_poo
    '\U0001F621',  # enraged_face
    '\U0001F62D',  # loudly_crying_face
    '\U0001F973',  # partying_face
    '\U0001F609',  # winking_face
    '\U0001F602',  # face_with_tears_of_joy
    '\U0001F605',  # grinning_face_with_sweat
    '\U0001F44D',  # thumbs_up
    '\U0001F607',  # smiling_face_with_halo
    '\U0001F373',  # cooking
    '\U0001F624',  # face_with_steam_from_nose
    '\U00002728',  # sparkles
    '\U0001F430',  # rabbit_face
    '\U0001F47B',  # ghost
    '\U0001F60E',  # smiling_face_with_sunglasses
    '\U0001F423',  # hatching_chick
    '\U0001F975',  # hot_face
    '\U0001F90D',  # white_heart
    '\U0001F389',  # party_popper
    '\U0001F480',  # skull
    '\U0001F914',  # thinking_face
    '\U0001F407',  # rabbit
    '\U0000263A',  # smiling_face
    '\U0001F60D',  # smiling_face_with_heart-eyes
    '\U0001F64F',  # folded_hands
    '\U0001F525',  # fire
    '\U0001F595',  # middle_finger
    '\U00002705',  # check_mark_button
    '\U0001F921'   # clown_face
]
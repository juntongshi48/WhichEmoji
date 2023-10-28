import re
import csv
import os
import glob

def extract_words(tweet):
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
    
    # Tokenize: Split the tweet into words
    words = tweet.split()
    
    # Further split words on the slash and flatten the list
    words = [item for sublist in [word.split('/') for word in words] for item in sublist]
    
    return words

def process_and_save_to_csv(input_filename, output_filename, specified_unicode):
    # Read the input CSV file
    with open(input_filename, 'r') as infile:
        reader = csv.reader(infile)
        # Skip header if any; if there's no header, comment the next line out
        next(reader)
        tweets = [row[0] for row in reader]

    # Process tweets
    processed_tweets = ["\t".join(extract_words(tweet)) for tweet in tweets]

    # Save to CSV
    with open(output_filename, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        # writer.writerow(["Specified Unicode", "Processed Tweet"])  # Writing the headers
        for processed_tweet in processed_tweets:
            writer.writerow([specified_unicode, processed_tweet])

def process_all_csvs_in_directory():
    input_directory = 'processed_data'
    output_directory = 'word_lists'
    
    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    i = 0

    # Loop over all CSV files in the input directory
    for csv_file in glob.glob(os.path.join(input_directory, '*.csv')):
        # Derive the output filename based on the input filename
        base_name = os.path.basename(csv_file)
        output_filename = os.path.join(output_directory, base_name)
        
        # Process the CSV file
        # print(csv_file)
        process_and_save_to_csv(csv_file, output_filename, emoji_unicode_list[i])
        i = i + 1

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
    '\U0001F62D',  # face_holding_back_tears
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

process_all_csvs_in_directory()
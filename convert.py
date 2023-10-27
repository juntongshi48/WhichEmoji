import pandas as pd
import re
import os
import glob

def remove_emojis(text):
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

def process_csv(input_filename, output_filename, specified_emoji):
    chunk_size = 5000  # Adjust this value based on your system's memory and file size
    df_list = []

    # Reading in chunks
    for chunk in pd.read_csv(input_filename, header=None, names=['text'], chunksize=chunk_size, engine='c'):
        # Filtering rows that contain the specified emoji
        chunk = chunk[chunk['text'].apply(lambda x: specified_emoji in str(x))]
        # Apply the emoji removal function
        chunk['text'] = chunk['text'].apply(remove_emojis)
        df_list.append(chunk)

    # Combining all chunks into a single DataFrame
    df = pd.concat(df_list, axis=0)

    # Saving the cleaned data to another CSV
    df.to_csv(output_filename, index=False, header=False)

# Load the CSV file into a DataFrame without headers
# input_filename = 'project_data/backhand_index_pointing_right.csv'
# output_filename = 'processed_data/backhand_index_pointing_right.csv'
# process_csv(input_filename, output_filename)

def process_all_csvs_in_directory():
    input_directory = 'project_data'
    output_directory = 'processed_data'
    
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
        process_csv(csv_file, output_filename, emoji_unicode_list[i])
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

# Call the function to process all CSV files in the directory
process_all_csvs_in_directory()


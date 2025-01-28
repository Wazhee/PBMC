import pandas as pd
import numpy as np
import os
from tqdm import tqdm

def save_df(df, src_dir):
    # Save the DataFrame to a CSV file
    df.to_csv(os.path.join(src_dir, "LOOCV.csv"), index=False)
    
def compile_csvfiles(csv_dir):
    df = pd.DataFrame()
    for i in range(65):
        # Read the CSV file into a temporary dataframe
        csvfile = os.path.join(csv_dir, f'{i}_slice.csv')
        temp_df = pd.read_csv(csvfile)
        # Append the temporary dataframe to the main dataframe
        df = pd.concat([df, temp_df], ignore_index=True)
        os.remove(csvfile) # clean folder when finished
    save_df(df, csv_dir)
    return df

def get_summary(df, col):
    scores = list(df[col])  # get cave
    mean = sum(scores) / len(scores) 
    variance = sum([((x - mean) ** 2) for x in scores]) / len(scores) 
    stdev = variance ** 0.5
    return round(mean, 2), round(stdev, 2)

def clean(src_dir):
    runs = [x for x in os.listdir(src_dir) if "EPOCHS300" in x]
    # Get averages for each column
    averages = {}
    for run in tqdm(runs): # iterate through every run completed
        csv_dir = os.path.join(src_dir, run)
        if "64_slice.csv" in os.listdir(csv_dir):
            try:
                df = compile_csvfiles(csv_dir)
                with open(os.path.join(csv_dir, "summary.txt"), "w") as file: # save summary of results
                    for col in list(df.columns):
                        mean, stdev = get_summary(df, col)
                        # Write to the file
                        file.write(f"AVERAGE {col.upper()}: {mean} || STDEV: {stdev}\n")
                        file.write("\n")
            except:
                print(f"'{csv_dir }' has missing files...")
if __name__ == "__main__":
    clean("../results/")
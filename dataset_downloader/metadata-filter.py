from argparse import ArgumentParser
import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--metadata', type=str, default='metadata.csv', help='metadata path')
    
    df = pd.read_csv('metadata.csv')

    train_df = df[df['partition'] == 'train'].sample(n=1000, random_state=42) 
    text_df = df[df['partition'] == 'text']

    final_df = pd.concat([text_df, train_df], ignore_index=True)
    final_df.to_csv('metadata_filtered.csv', index=False)
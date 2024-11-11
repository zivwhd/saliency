import os
import glob
import pickle
import pandas as pd
import sys

def read_stats(base_path, model_name):
    data = []

    # Iterate over files matching the pattern base_path/*/ILS*
    for file_path in glob.glob(os.path.join(base_path, '*/ILS*')):
        method = os.path.basename(os.path.dirname(file_path))  # Get the method from dirname
        filename = os.path.basename(file_path)

        # Load stats from pickle file
        print(f">> loading {file_path}")
        with open(file_path, 'rb') as f:
            stats = pickle.load(f)

        # Flatten into a dictionary with model_name, method, filename, and stats
        row = {'model_name': model_name, 'image': filename, 'variant': method, **stats}
        data.append(row)

    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    return df

def main():
    # Read model_name, base_path, and output_path from command line arguments
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_name> <base_path> <output_path>")
        sys.exit(1)
    
    model_name = sys.argv[1]
    base_path = sys.argv[2]
    output_path = sys.argv[3]

    # Generate the DataFrame
    df = read_stats(base_path, model_name)

    # Save DataFrame to CSV
    #df.to_csv(output_path, index=False)
    df.to_pickle(output_path)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    main()

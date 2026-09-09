import pandas as pd
import sys
import os

def convert_labels(csv_path):

    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] < 2:
        print("Error: The CSV needs to have two columns as (file, classification).")
        return

    def map_value(x):
        if x == "left_curve":
            return "-30°"
        elif x == "right_curve":
            return "30°"
        else:
            return "0°"

    df.iloc[:, 1] = df.iloc[:, 1].apply(map_value)

    base, ext = os.path.splitext(csv_path)
    output_path = base + "_converted.csv"

    df.to_csv(output_path, header=False, index=False)

    print(f"Salved file in: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Use: python convert_labels.py caminho_para_arquivo.csv")
    else:
        convert_labels(sys.argv[1])

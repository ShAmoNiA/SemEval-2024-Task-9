import csv
import json
import os

def csv_to_json(csv_file_path):
    with open(csv_file_path, 'r') as csv_file:
        csv_data = csv.DictReader(csv_file)
        data_list = []
        for row in csv_data:
            data_list.append(row)
            
    return data_list

def convert_all_csv_to_json(csv_folder_path):
    listHold = []
    # Loop through all CSV files in the specified folder
    for csv_file in os.listdir(csv_folder_path):
        if csv_file.endswith('.csv'):
            csv_file_path = os.path.join(csv_folder_path, csv_file)
            json_file_path = os.path.join(csv_folder_path, f'{os.path.splitext(csv_file)[0]}.csv')
            listHold.append(csv_to_json(json_file_path))
    print(listHold)


# Example usage
csv_folder_path = 'Datas'
convert_all_csv_to_json(csv_folder_path)

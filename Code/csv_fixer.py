import csv


file_path = 'main.csv'
List_hold = []

with open(file_path, 'r', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        List_hold.append(row)

for i in range(len(List_hold)):
    holdA = List_hold[i][2].split('\n')
    for item in holdA:
        item = item.replace('option : ','').replace('option 1:','').replace('option 2:','').replace('option 3:','')
        print(item)

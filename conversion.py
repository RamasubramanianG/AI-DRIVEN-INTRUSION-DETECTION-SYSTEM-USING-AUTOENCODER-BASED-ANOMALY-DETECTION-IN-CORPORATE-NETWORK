import arff
import csv

# Load the ARFF file
with open(r'E:\project\new_project\project_implementationv2\data\NSL_KDD\KDDTrain+.arff','r') as f:
    dataset = arff.load(f)

# Extract attributes and data
attributes = [attr[0] for attr in dataset['attributes']]
data = dataset['data']

# Write to CSV
with open('output_file.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(attributes)  # Write header
    writer.writerows(data)       # Write data

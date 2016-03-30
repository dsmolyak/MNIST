import csv

ofile  = open('/home/smolydb1/Projects/MNIST/accuracy_data.csv', "wb")
writer = csv.writer(ofile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
row = ["Precision","Recall","f1-score"]
writer.writerow(row)


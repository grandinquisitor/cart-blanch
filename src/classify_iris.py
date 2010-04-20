import csv
from tree import *

discriminators = feature_collection(
    numeric_feature('sepal_length'),
    numeric_feature('sepal_width'), 
    numeric_feature('petal_length'),
    numeric_feature('petal_width')
)


def get_rows():
    datareader = csv.reader(open('fisher_iris.txt'))

    for datarow in datareader:
        sepal_length, sepal_width, petal_length, petal_width = map(float, datarow[:4])
        class_label = datarow[4]

        yield {'sepal_length': sepal_length,
               'sepal_width': sepal_width,
               'petal_length': petal_length,
               'petal_width': petal_width,
               '__CLASS__': class_label}

training_set = discriminators.build_training_set(get_rows())

tree = discriminators.buildtree(training_set)

tree.printme()

tree.prune(.4);

#tree.print_python_classifier('classify')

tree.print_mysql_classifier('classify_iris')

print tree.expected_error()
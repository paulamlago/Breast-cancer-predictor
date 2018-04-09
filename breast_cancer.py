from sklearn.datasets import load_breast_cancer 
from sklearn import tree
import numpy
import sys
from colorama import init
init(strip=not sys.stdout.isatty()) # strip colors if stdout is redirected
from pyfiglet import figlet_format

print(figlet_format('Breast cancer predictor', font='big', width = 1000))
print(figlet_format('by Paulamlago'))
print("Please answer the questions with the correct number or any other character if you don't know that information")
pacient_data = []

bc = load_breast_cancer()

medias = []
#establecer las medias de cada feature para que si el usuario no sabe su dato, rellenarlo con la media de todos
for i in range(len(bc.feature_names)):
    column = bc.data[:,i]
    sum = 0
    for j in range(len(column)):
        sum += column[j]
    medias.append(sum / len(column))



for i in range(len(bc.feature_names)):
    try:
        float_input = True
        data = float(input('Which is the ' + bc.feature_names[i] + '?: '))
    except ValueError:
        data = medias[i]
        print("Let's use the number: ", data)
    
        
    pacient_data.append(data)

#creamos nuestro arbol de decision y lo rellenamos con los datos validos
clf = tree.DecisionTreeClassifier()
clf = clf.fit(bc.data, bc.target)

result = clf.predict([pacient_data])
print("Based on other pacients, the system has predicted the followign: ")
if(result is 0):
    print("Sorry, your tumor is malignant")
else:
    print("You're lucky, your tumor is benign")
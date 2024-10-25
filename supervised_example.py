import pandas
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree




if __name__ == "__main__":
    """
    Main of the data analysis
    """
    
    #0 load dataset (library PANDAS; associated with it there is NUMPY)
    my_dataset = pandas.read_csv("./labelled_dataset.csv")
    label_obj = my_dataset["label"]
    data_obj = my_dataset.drop(columns=["label", "time", "datetime"])
    # the row above is "equivalent" to
    # my_dataset.drop(columns=["label"], inplace=True)

    #1 split dataset
    train_data, train_label, test_data, test_label = train_test_split(data_obj, label_obj, test_size=0.5)

    #2 choose classifier (library SCIKIT LEARN)
    clf = tree.DecisionTreeClassifier()

    #3 train classifier
    clf = clf.fit(train_data, train_label)

    #4 test classifier
    predicted_labels = clf.predict(test_data)
    acc_score = accuracy_score(test_label, predicted_labels)
    print("Accuracy is " + str(acc_score))
    print("Accuracy is %.3f" % acc_score)

    tn, tp, fn, tp = confusion_matrix(test_data, predicted_labels).ravel()
    print("TP: %d, TN: %d, FN: %d, FP: %d" % (tp, tn, fn, tp))

    a = 1
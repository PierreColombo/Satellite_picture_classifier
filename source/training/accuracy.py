import sklearn


# The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches
# # its best value at 1 and worst score at 0

# y_pred estimated targets
# y groundtruth
def accuracy(y,y_pred) :
    return sklearn.metrics.f1_score(y, y_pred, average='weighted')

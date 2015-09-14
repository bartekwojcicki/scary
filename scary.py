import argparse
import git
from sklearn import cross_validation
from sklearn import naive_bayes
from sklearn import metrics

from scary import dataset
from scary import evaluation


def run():
    arguments = parse_arguments()
    repository = git.Repo(arguments.repository)
    if arguments.evaluate:
        evaluate(repository, arguments.train_from_revision, arguments.train_to_revision)
    else:
        predict(repository, arguments.revision, arguments.train_from_revision, arguments.train_to_revision)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Fault predictor for Python projects.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--evaluate', help="run evaluation instead of predicting", action='store_true')
    parser.add_argument('repository', help="path to repository")
    parser.add_argument('--train_from_revision', help="build training set from revision", default='HEAD~100')
    parser.add_argument('--train_to_revision', help="build training set to revision", default='HEAD')
    parser.add_argument('--revision', help="predict for revision", default='HEAD')
    return parser.parse_args()


def predict(repository, predict_revision, train_from_revision, train_to_revision):
    predicting_set = dataset.PredictingSetBuilder.build_predicting_set(
        repository, predict_revision)
    training_set = dataset.TrainingSetBuilder.build_training_set(
        repository, train_from_revision, train_to_revision)
    classifier = naive_bayes.GaussianNB()
    classifier.fit(training_set.features, training_set.classes)
    classes = classifier.predict(predicting_set.features)
    predictions = zip(predicting_set.functions, classes)
    print("FAULT-PRONE FUNCTIONS:")
    for prediction in predictions:
        if prediction[1]:
            function = prediction[0]
            print('{}:{} {}'.format(function.file, function.lineno, function.name))


def evaluate(repository, train_from_revision, train_to_revision):
    training_set = dataset.TrainingSetBuilder.build_training_set(
        repository, train_from_revision, train_to_revision)
    classifier = naive_bayes.GaussianNB()
    predictions = cross_validation.cross_val_predict(
        classifier, training_set.features, training_set.classes, cv=10)
    confusion_matrix = evaluation.ConfusionMatrix(
        metrics.confusion_matrix(training_set.classes, predictions))
    print('RECALL: {}'.format(evaluation.recall(confusion_matrix)))
    print('FALSE_POSITIVE_RATE: {}'.format(
        evaluation.false_positive_rate(confusion_matrix)))


if __name__ == '__main__':
    run()

import collections

from . import fetchers


PredictingSet = collections.namedtuple(
    'PredictingSet', ['functions', 'features'])


class PredictingSetBuilder:
    def __init__(self, *, repository, functions_fetcher,
                 features_fetcher):
        self.repository = repository
        self.functions_fetcher = functions_fetcher
        self.features_fetcher = features_fetcher

    def build(self):
        functions = self.functions
        features = self.get_features(functions)
        return PredictingSet(functions=functions, features=features)

    @property
    def functions(self):
        return self.functions_fetcher.fetch()

    def get_features(self, functions):
        return self.features_fetcher.fetch(functions)

    @classmethod
    def build_predicting_set(cls, repository, revision):
        files_fetcher = fetchers.FilesFetcher(
            repository=repository, revision=revision)
        functions_fetcher = fetchers.FunctionsFetcher(
            repository=repository,
            from_revision=revision,
            files_fetcher=files_fetcher)
        features_fetcher = fetchers.FeaturesFetcher(
            repository=repository,
            revision=revision)
        builder = cls(
            repository=repository,
            functions_fetcher=functions_fetcher,
            features_fetcher=features_fetcher)
        return builder.build()


TrainingSet = collections.namedtuple(
    'TrainingSet', ['functions', 'features', 'classes'])


class TrainingSetBuilder:
    def __init__(self, *, repository, functions_fetcher, features_fetcher,
                 classes_fetcher):
        self.repository = repository
        self.functions_fetcher = functions_fetcher
        self.features_fetcher = features_fetcher
        self.classes_fetcher = classes_fetcher

    def build(self):
        functions = self.functions
        features = self.get_features(functions)
        classes = self.get_classes(functions)
        return TrainingSet(
            functions=functions,
            features=features,
            classes=classes)

    @property
    def functions(self):
        return self.functions_fetcher.fetch()

    def get_features(self, functions):
        return self.features_fetcher.fetch(functions)

    def get_classes(self, functions):
        return self.classes_fetcher.fetch(functions)

    @classmethod
    def build_training_set(cls, repository, from_revision, to_revision):
        files_fetcher = fetchers.ModifiedFilesFetcher(
            repository=repository,
            from_revision=from_revision,
            to_revision=to_revision)
        functions_fetcher = fetchers.UniqueFunctionsFetcher(
            repository=repository,
            from_revision=from_revision,
            to_revision=to_revision,
            files_fetcher=files_fetcher)
        features_fetcher = fetchers.FeaturesFetcher(
            repository=repository,
            revision=from_revision)
        classes_fetcher = fetchers.ClassesFetcher(
            repository=repository,
            from_revision=from_revision,
            to_revision=to_revision)
        builder = cls(
            repository=repository,
            functions_fetcher=functions_fetcher,
            features_fetcher=features_fetcher,
            classes_fetcher=classes_fetcher)
        return builder.build()

import collections

import git
import numpy as np
from radon import visitors

from scary import halstead


class FunctionsFetcher:
    def __init__(
            self, *, repository, from_revision, files_fetcher, to_revision=None):
        self.repository = repository
        self.from_revision = from_revision
        self.to_revision = to_revision
        self.files_fetcher = files_fetcher

    def fetch_unique(self):
        functions_dict = {}
        for function in self.fetch():
            function_key = (function.name, function.file)
            if function_key not in functions_dict:
                functions_dict[function_key] = function
        return list(functions_dict.values())

    def fetch(self):
        functions = self.get_functions_for_revision(self.from_revision)
        if self.to_revision:
            return list(self.filter_functions(functions))
        else:
            return list(functions)

    def get_functions_for_revision(self, revision):
        for file in self.files:
            yield from self.get_functions_for_file_and_revision(file, revision)

    @property
    def files(self):
        return self.files_fetcher.files

    def get_functions_for_file_and_revision(self, file, revision):
        try:
            code = get_code(
                git=self.repository.git, revision=revision, file=file)
            visitor = FunctionsVisitor.from_code(code, file=file)
            yield from visitor.functions
        except (git.exc.GitCommandError, SyntaxError):
            pass

    def filter_functions(self, functions):
        allowed_functions = self.get_functions_set(self.to_revision)
        for function in functions:
            if (function.name, function.file) in allowed_functions:
                yield function

    def get_functions_set(self, revision):
        functions = self.get_functions_for_revision(revision)
        return {(function.name, function.file) for function in functions}


class FilesFetcher:
    def __init__(self, *, repository, revision):
        self.repository = repository
        self.revision = revision

    def fetch(self):
        return list(self.files)

    @property
    def files(self):
        all_files = self.all_files
        return self.filter_files(all_files)

    @property
    def all_files(self):
        all_files_string = self.git.ls_tree('-r', '--name-only', self.revision)
        return all_files_string.split('\n')

    @property
    def git(self):
        return self.repository.git

    @staticmethod
    def filter_files(files):
        for file in files:
            if file.endswith('.py') and 'docs' not in file:
                yield file


class ModifiedFilesFetcher(FilesFetcher):
    def __init__(self, *, repository, from_revision, to_revision):
        super().__init__(repository=repository, revision=from_revision)
        self.to_revision = to_revision

    @property
    def all_files(self):
        files_with_status_string = self.git.diff(
            '--name-status', self.revision, self.to_revision)
        for file_with_status in files_with_status_string.split('\n'):
            status, file = file_with_status.split()
            if status == 'M':
                yield file


class FunctionsVisitor(visitors.CodeVisitor):
    def __init__(self, file=''):
        self.functions = []
        self.file = file

    def visit_FunctionDef(self, node):
        self.functions.append(Function(node.name, self.file, node.lineno))
        for child in node.body:
            self.visit(child)


Function = collections.namedtuple('Function', ['name', 'file', 'lineno'])


class FeaturesFetcher:
    def __init__(self, repository, revision):
        self.repository = repository
        self.revision = revision

    def fetch(self, functions):
        return np.array(list(self.get_features(functions)))

    def get_features(self, functions):
        files = self.get_files(functions)
        features = self.get_features_for_files(files)
        features_dict = self.get_features_by_functions(features)
        for function in functions:
            yield self.get_features_for_function(features_dict, function)

    @staticmethod
    def get_files(functions):
        return {function.file for function in functions}

    def get_features_for_files(self, files):
        for file in files:
            yield from self.get_features_for_file(file)

    def get_features_for_file(self, file):
        code = get_code(
            git=self.repository.git, revision=self.revision, file=file)
        code_lines = code.split('\n')
        visitor = FeaturesVisitor.from_code(code, file=file, code_lines=code_lines)
        yield from visitor.features

    @staticmethod
    def get_features_by_functions(features):
        return {
            (feature.function_name, feature.file, feature.lineno): feature.features
            for feature in features
        }

    @staticmethod
    def get_features_for_function(features_dict, function):
        return features_dict[(function.name, function.file, function.lineno)]


class FeaturesVisitor(visitors.CodeVisitor):
    def __init__(self, file='', code_lines=[]):
        self.features = []
        self.file = file
        self.code_lines = code_lines

    def visit_FunctionDef(self, node):
        complexity_visitor = visitors.ComplexityVisitor.from_ast(node)
        start_line = node.lineno - 1
        end_line = complexity_visitor.functions[0].endline
        halstead_metrics = halstead.HalsteadMetricsCollector().try_collect(
            self.code_lines, start_line, end_line)
        features = [
            halstead_metrics.h1,
            halstead_metrics.h2,
            halstead_metrics.N1,
            halstead_metrics.N2,
            halstead_metrics.vocabulary,
            halstead_metrics.length,
            halstead_metrics.calculated_length,
            halstead_metrics.volume,
            halstead_metrics.difficulty,
            halstead_metrics.effort,
            halstead_metrics.time,
            halstead_metrics.bugs,
            complexity_visitor.total_complexity,
            complexity_visitor.complexity,
            complexity_visitor.functions_complexity,
            complexity_visitor.classes_complexity,
        ]
        self.features.append(Feature(node.name, self.file, node.lineno, features))
        for child in node.body:
            self.visit(child)


Feature = collections.namedtuple(
    'Feature', ['function_name', 'file', 'lineno', 'features'])


def get_code(*, git, revision, file):
    return git.show('{}:{}'.format(revision, file))


class ClassesFetcher:
    def __init__(self, repository, from_revision, to_revision):
        self.repository = repository
        self.from_revision = from_revision
        self.to_revision = to_revision

    def fetch(self, functions):
        return np.array(list(self.get_classes(functions)))

    def get_classes(self, functions):
        for function in functions:
            yield self.get_class_for_function(function)

    def get_class_for_function(self, function):
        trace_lines = ':{}:{}'.format(function.name, function.file)
        revision_range = '{}..{}'.format(self.from_revision, self.to_revision)
        fixes = self.git.log('-L', trace_lines, '--grep=fix', '--grep=bug',
                             '--grep=issue', '--regexp-ignore-case',
                             revision_range)
        return bool(fixes)

    @property
    def git(self):
        return self.repository.git

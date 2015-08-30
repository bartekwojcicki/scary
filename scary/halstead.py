import io
import keyword
import math
import tokenize

from radon import metrics


class HalsteadMetricsCollector:
    OPERATOR_TYPES = [
        tokenize.OP,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
    ]

    OPERAND_TYPES = [
        tokenize.STRING,
        tokenize.NUMBER,
    ]

    def collect(self, tokens):
        operators, operands = self.count_tokens(tokens)
        return self.summary(operators, operands)

    def count_tokens(self, tokens):
        operators = []
        operands = []
        for token in tokens:
            if self.is_operator(token):
                operators.append(token.string)
            elif self.is_operand(token):
                operands.append(token.string)
        return operators, operands

    def is_operator(self, token):
        return (
            token.type in self.OPERATOR_TYPES or
            (token.type == tokenize.NAME and keyword.iskeyword(token.string))
        )

    def is_operand(self, token):
        return (
            token.type in self.OPERAND_TYPES or
            (token.type == tokenize.NAME and not keyword.iskeyword(token.string))
        )

    def summary(self, operators, operands):
        h1 = len(set(operators))
        h2 = len(set(operands))
        N1 = len(operators)
        N2 = len(operands)
        h = h1 + h2
        N = N1 + N2
        if h1 and h2:
            length = h1 * math.log(h1, 2) + h2 * math.log(h2, 2)
        else:
            length = 0
        volume = N * math.log(h, 2) if h != 0 else 0
        difficulty = (h1 * N2) / float(2 * h2) if h2 != 0 else 0
        effort = difficulty * volume
        return metrics.Halstead(
            h1, h2, N1, N2, h, N, length, volume, difficulty, effort,
            effort / 18., volume / 3000.
        )

    def collect_from_code(self, code):
        bytes_io = io.BytesIO(bytes(code, encoding='utf8'))
        tokens = tokenize.tokenize(bytes_io.readline)
        return self.collect(tokens)

    def try_collect(self, code_lines, start_line, end_line):
        while True:
            try:
                code = '\n'.join(code_lines[start_line:end_line])
                return self.collect_from_code(code)
            except tokenize.TokenError:
                end_line += 1

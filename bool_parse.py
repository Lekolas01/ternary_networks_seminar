import collections
import re

from bool_formula import AND, OR, Bool, Literal, possible_data


class ExpressionEvaluator:
    """
    Implementation of a recursive descent parser for boolean expressions.
    """

    def __init__(self):
        self.LIT_TOKEN = r"(?P<LIT_TOKEN>[\w\d]+)"
        self.AND_TOKEN = r"(?P<AND_TOKEN>\&)"
        self.OR_TOKEN = r"(?P<OR_TOKEN>\|)"
        self.LPAREN_TOKEN = r"(?P<LPAREN_TOKEN>\()"
        self.RPAREN_TOKEN = r"(?P<RPAREN_TOKEN>\))"
        self.WS_TOKEN = r"(?P<WS_TOKEN>\s+)"

        self.master_pattern = re.compile(
            "|".join(
                (
                    self.LPAREN_TOKEN,
                    self.LIT_TOKEN,
                    self.AND_TOKEN,
                    self.OR_TOKEN,
                    self.RPAREN_TOKEN,
                    self.WS_TOKEN,
                )
            )
        )
        self.Token = collections.namedtuple("Token", ["type", "value"])

    def generate_tokens(self, pattern, text):
        scanner = pattern.scanner(text)
        for m in iter(scanner.match, None):
            token = self.Token(m.lastgroup, m.group())

            if token.type != "WS_TOKEN":
                yield token

    def parse(self, text):
        self.tokens = self.generate_tokens(self.master_pattern, text)
        self.current_token = None
        self.next_token = None
        self._advance()

        # expr is the top level grammar. So we invoke that first.
        # it will invoke other function to consume tokens according to grammar rule.
        return self.expr()

    def _advance(self):
        self.current_token, self.next_token = self.next_token, next(self.tokens, None)

    def _accept(self, token_type):
        # if there is next token and token type matches
        if self.next_token and self.next_token.type == token_type:
            self._advance()
            return True
        else:
            return False

    def _expect(self, token_type):
        if not self._accept(token_type):
            raise SyntaxError("Expected " + token_type)

    def expr(self) -> Bool:
        """
        expr := rule { OR rule }
        """
        rules = []
        rules.append(self.rule())

        while self._accept("OR_TOKEN"):
            rules.append(self.rule())
        if len(rules) == 1:
            return rules[0]
        return OR(*rules)

    def rule(self) -> Bool:
        """
        rule := conjunct { AND conjunct }
        """
        conjuncts = []
        conjuncts.append(self.conjunct())

        while self._accept("AND_TOKEN"):
            conjuncts.append(self.conjunct())

        if len(conjuncts) == 1:
            return conjuncts[0]
        return AND(*conjuncts)

    def conjunct(self) -> Bool:
        """
        conjunct := LIT | LPARAM expr RPARAM
        """
        if self._accept("LIT_TOKEN"):
            return Literal(self.current_token.value)  # type:ignore
        elif self._accept("LPAREN_TOKEN"):
            expr = self.expr()
            self._expect("RPAREN_TOKEN")
            return expr
        else:
            raise SyntaxError("Expect NUMBER or LPAREN")


if __name__ == "__main__":
    e = ExpressionEvaluator()
    expr = e.parse("(g | (f & (e | (d & (c | (b & a))))))")
    print(expr)

    data = possible_data(expr.all_literals())
    print(expr(data))

from __future__ import annotations

import re
from dataclasses import dataclass

from findex.index import Index, intersect_merge, union_merge
from findex.tokenize import tokenize


class QuerySyntaxError(ValueError):
    """Raised when a user search query has invalid syntax."""


class QueryNode:
    def evaluate(self, index: Index) -> set[int]:
        raise NotImplementedError

    def terms(self) -> list[str]:
        raise NotImplementedError

    def __and__(self, other: QueryNode) -> And:
        return And(self, other)

    def __or__(self, other: QueryNode) -> Or:
        return Or(self, other)

    def __invert__(self) -> Not:
        return Not(self)


@dataclass(frozen=True, slots=True)
class Term(QueryNode):
    value: str

    def evaluate(self, index: Index) -> set[int]:
        plist = index.postings.get(self.value)
        if not plist:
            return set()
        return {p.doc_id for p in plist}

    def terms(self) -> list[str]:
        return [self.value]


@dataclass(frozen=True, slots=True)
class Phrase(QueryNode):
    words: tuple[str, ...]

    def evaluate(self, index: Index) -> set[int]:
        if not self.words:
            return set()
        if len(self.words) == 1:
            return Term(self.words[0]).evaluate(index)
        if not index.with_positions:
            raise QuerySyntaxError(
                "Phrase queries require positional postings; rebuild index with --positions"
            )

        postings_lists = [index.postings.get(w, []) for w in self.words]
        if any(not pl for pl in postings_lists):
            return set()

        per_term_maps = [{p.doc_id: p.positions for p in pl} for pl in postings_lists]
        candidate_docs = set(per_term_maps[0].keys())
        for m in per_term_maps[1:]:
            candidate_docs &= set(m.keys())

        matched: set[int] = set()
        for doc_id in candidate_docs:
            valid_starts = set(per_term_maps[0][doc_id])
            for offset, pos_map in enumerate(per_term_maps[1:], start=1):
                next_positions = set(pos_map[doc_id])
                valid_starts = {p for p in valid_starts if (p + offset) in next_positions}
                if not valid_starts:
                    break
            if valid_starts:
                matched.add(doc_id)
        return matched

    def terms(self) -> list[str]:
        return list(self.words)


@dataclass(frozen=True, slots=True)
class And(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        lhs = sorted(self.left.evaluate(index))
        rhs = sorted(self.right.evaluate(index))
        return set(intersect_merge(lhs, rhs))

    def terms(self) -> list[str]:
        return self.left.terms() + self.right.terms()


@dataclass(frozen=True, slots=True)
class Or(QueryNode):
    left: QueryNode
    right: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        lhs = sorted(self.left.evaluate(index))
        rhs = sorted(self.right.evaluate(index))
        return set(union_merge(lhs, rhs))

    def terms(self) -> list[str]:
        return self.left.terms() + self.right.terms()


@dataclass(frozen=True, slots=True)
class Not(QueryNode):
    operand: QueryNode

    def evaluate(self, index: Index) -> set[int]:
        all_docs = set(index.doc_meta.keys())
        return all_docs - self.operand.evaluate(index)

    def terms(self) -> list[str]:
        return []


_TOKEN_SPEC = re.compile(
    r"""
    \s*(?:
        (?P<LPAREN>\()
      | (?P<RPAREN>\))
      | (?P<PHRASE>"[^"]*")
      | (?P<WORD>[^\s()"]+)
    )
    """,
    re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class _Tok:
    kind: str
    value: str


def _lex_query(raw: str) -> list[_Tok]:
    tokens: list[_Tok] = []
    pos = 0
    while pos < len(raw):
        match = _TOKEN_SPEC.match(raw, pos)
        if not match:
            raise QuerySyntaxError(f"Unexpected character at position {pos}: {raw[pos:]!r}")
        pos = match.end()
        kind = match.lastgroup
        if kind is None:
            continue
        val = match.group(kind)
        if kind == "WORD":
            upper = val.upper()
            if upper in {"AND", "OR", "NOT"}:
                tokens.append(_Tok(upper, upper))
            else:
                for t in tokenize(val):
                    tokens.append(_Tok("TERM", t))
        elif kind == "PHRASE":
            inner = val[1:-1]
            words = tuple(tokenize(inner))
            if not words:
                raise QuerySyntaxError("Empty quoted phrase in query")
            tokens.append(_Tok("PHRASE", " ".join(words)))
        else:
            tokens.append(_Tok(kind, val))
    return tokens


class _Parser:
    def __init__(self, tokens: list[_Tok]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> _Tok | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def consume(self) -> _Tok:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self) -> QueryNode:
        if not self.tokens:
            raise QuerySyntaxError("Query cannot be empty")
        node = self._parse_or()
        if self.peek() is not None:
            raise QuerySyntaxError(f"Unexpected token {self.peek()!r}")
        return node

    def _parse_or(self) -> QueryNode:
        node = self._parse_and()
        while (tok := self.peek()) is not None and tok.kind == "OR":
            self.consume()
            rhs = self._parse_and()
            node = Or(node, rhs)
        return node

    def _parse_and(self) -> QueryNode:
        node = self._parse_not()
        while (tok := self.peek()) is not None:
            if tok.kind == "AND":
                self.consume()
                rhs = self._parse_not()
                node = And(node, rhs)
            elif tok.kind in {"TERM", "PHRASE", "NOT", "LPAREN"}:
                rhs = self._parse_not()
                node = And(node, rhs)
            else:
                break
        return node

    def _parse_not(self) -> QueryNode:
        tok = self.peek()
        if tok is not None and tok.kind == "NOT":
            self.consume()
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> QueryNode:
        tok = self.peek()
        if tok is None:
            raise QuerySyntaxError("Unexpected end of query")
        if tok.kind == "TERM":
            self.consume()
            return Term(tok.value)
        if tok.kind == "PHRASE":
            self.consume()
            words = tuple(tok.value.split())
            if len(words) == 1:
                return Term(words[0])
            return Phrase(words)
        if tok.kind == "LPAREN":
            self.consume()
            node = self._parse_or()
            closing = self.peek()
            if closing is None or closing.kind != "RPAREN":
                raise QuerySyntaxError("Unclosed '(' in query")
            self.consume()
            return node
        raise QuerySyntaxError(f"Unexpected token {tok!r}")


def parse_query(raw: str) -> QueryNode:
    tokens = _lex_query(raw)
    return _Parser(tokens).parse()

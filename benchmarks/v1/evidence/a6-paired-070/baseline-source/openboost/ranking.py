"""Query-local CPU ranking geometry; pair dependencies reduce into ordinary row fields."""

from dataclasses import dataclass

import numpy as np

from .data import Problem, _owned


@dataclass(frozen=True, eq=False)
class PairGeometry:
    loss: float
    gradient: np.ndarray
    curvature: np.ndarray


@dataclass(frozen=True)
class Ranking:
    """All eligible pairs per query, normalized by eligible pair count.

    Lambda delta-NDCG weights are frozen for each geometry call. They are not
    differentiated. Query weights are repeated aligned roles, not row weights.
    """

    lambdas: bool = False
    k: int = 10

    def __post_init__(self):
        if type(self.lambdas) is not bool or type(self.k) is not int or self.k < 1:
            raise ValueError("boolean lambdas and positive integer k required")

    @staticmethod
    def validate(problem):
        if (
            not isinstance(problem, Problem)
            or problem.classes is not None
            or problem.target.shape[1] != 1
            or problem.raw_width != 1
            or "query" not in problem.structure
            or not set(problem.structure) <= {"query", "query_weight"}
        ):
            raise ValueError("ranking requires scalar relevance and explicit query roles")
        if np.any(problem.weight != 1):
            raise ValueError("ranking requires query weights, not row weights")
        relevance = problem.target[:, 0]
        query = problem.structure["query"]
        if (
            query.shape != problem.target.shape
            or np.any(query != np.floor(query))
            or np.any(np.abs(query) > 2**53 - 1)
            or np.any(relevance < 0)
            or np.any(relevance != np.floor(relevance))
        ):
            raise ValueError("integer query codes and nonnegative integer relevance required")
        weights = problem.structure.get("query_weight", np.ones_like(query))
        if weights.shape != query.shape or np.any(weights < 0):
            raise ValueError("aligned nonnegative query weights required")
        for q in np.unique(query):
            w = weights[query[:, 0] == q, 0]
            if np.any(w != w[0]):
                raise ValueError("query weight must be constant within each query")
        if not np.any(weights > 0):
            raise ValueError("at least one positive query weight required")

    def _groups(self, problem, raw):
        self.validate(problem)
        scores = problem.with_offset(raw)[:, 0]
        query = problem.structure["query"][:, 0]
        weights = problem.structure.get("query_weight", np.ones_like(problem.target))[:, 0]
        with np.errstate(over="raise", invalid="raise"):
            gains = np.exp2(problem.target[:, 0]) - 1
            for q in np.unique(query):
                rows = np.flatnonzero(query == q)
                discount = np.zeros(len(rows))
                end = min(self.k, len(rows))
                discount[:end] = 1 / np.log2(np.arange(end) + 2)
                ideal = np.dot(np.sort(gains[rows])[::-1], discount)
                if not np.isfinite(ideal):
                    raise ValueError("nonfinite ideal DCG")
                order = np.lexsort((problem.row_ids[rows], -scores[rows]))
                ranked_discount = np.empty(len(rows))
                ranked_discount[order] = discount
                yield rows, scores[rows], gains[rows], ranked_discount, ideal, weights[rows[0]]

    def geometry(self, problem, raw):
        gradient = np.zeros(len(problem.target))
        curvature = np.zeros_like(gradient)
        loss = 0.0
        for rows, scores, gains, discount, ideal, weight in self._groups(problem, raw):
            relevance = problem.target[rows, 0]
            high, low = np.nonzero(relevance[:, None] > relevance[None, :])
            if not len(high):
                continue
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                factor = np.full(len(high), weight / len(high))
                if self.lambdas:
                    factor *= (
                        np.abs((gains[high] - gains[low]) * (discount[high] - discount[low]))
                        / ideal
                        if ideal > 0
                        else 0
                    )
                difference = scores[high] - scores[low]
                tail = np.exp(-np.abs(difference))
                probability = np.where(difference >= 0, tail / (1 + tail), 1 / (1 + tail))
                h = factor * tail / (1 + tail) ** 2
                g = factor * probability
                loss += float(np.dot(factor, np.logaddexp(0, -difference)))
                np.add.at(gradient, rows[high], -g)
                np.add.at(gradient, rows[low], g)
                np.add.at(curvature, rows[high], h)
                np.add.at(curvature, rows[low], h)
        if not np.isfinite(loss):
            raise ValueError("nonfinite pair loss")
        return PairGeometry(loss, _owned(gradient, ndim=1), _owned(curvature, ndim=1))

    def score(self, problem, raw):
        """One minus query-weighted mean NDCG@k; smaller is better."""
        values, weights = [], []
        for _rows, _scores, gains, discount, ideal, weight in self._groups(problem, raw):
            values.append(1.0 if ideal == 0 else float(np.dot(gains, discount) / ideal))
            weights.append(weight)
        weights = np.asarray(weights)
        # Scale before summing to avoid overflowing a finite query-weight vector.
        weights = weights / weights.max()
        return float(1 - np.dot(weights / weights.sum(), values))

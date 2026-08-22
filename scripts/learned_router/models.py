"""Pure-Python ordinal logistic and shallow boosted-tree routers."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .features import FeatureEncoder


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _logit(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


def _score(probability_balanced: float, probability_strongest: float) -> float:
    probability_strongest = min(probability_strongest, probability_balanced)
    return min(100.0, max(0.0, 10.0 + 45.0 * probability_balanced + 35.0 * probability_strongest))


def tier_for_score(score: float) -> str:
    if score >= 75.0:
        return "strongest"
    if score >= 35.0:
        return "balanced"
    return "economical"


@dataclass(frozen=True)
class Prediction:
    score: float
    probability_balanced: float
    probability_strongest: float
    tier: str


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Solve a small dense linear system with pivoted Gaussian elimination."""

    size = len(vector)
    augmented = [row[:] + [vector[index]] for index, row in enumerate(matrix)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] = 1e-12
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for entry in range(column, size + 1):
            augmented[column][entry] /= divisor
        for row in range(size):
            if row == column:
                continue
            multiplier = augmented[row][column]
            if multiplier == 0:
                continue
            for entry in range(column, size + 1):
                augmented[row][entry] -= multiplier * augmented[column][entry]
    return [augmented[row][-1] for row in range(size)]


def _weighted_log_loss(
    x_rows: Sequence[Sequence[float]],
    targets: Sequence[int],
    coefficients: Sequence[float],
    positive_weight: float,
    l2: float,
) -> float:
    loss = 0.0
    for row, target in zip(x_rows, targets):
        value = coefficients[0] + sum(
            coefficient * feature
            for coefficient, feature in zip(coefficients[1:], row)
        )
        example_weight = positive_weight if target else 1.0
        loss += example_weight * (max(value, 0.0) - target * value + math.log1p(math.exp(-abs(value))))
    loss += 0.5 * l2 * sum(value * value for value in coefficients[1:])
    return loss


def _fit_logistic_head(
    x_rows: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    positive_weight: float,
    l2: float,
    max_iterations: int,
) -> list[float]:
    dimension = len(x_rows[0])
    weighted_positives = positive_weight * sum(targets)
    weighted_total = weighted_positives + len(targets) - sum(targets)
    coefficients = [_logit(weighted_positives / weighted_total)] + [0.0] * dimension

    for _ in range(max_iterations):
        gradient = [0.0] * (dimension + 1)
        hessian = [[0.0] * (dimension + 1) for _ in range(dimension + 1)]
        for row, target in zip(x_rows, targets):
            expanded = [1.0, *row]
            value = sum(weight * feature for weight, feature in zip(coefficients, expanded))
            probability = _sigmoid(value)
            example_weight = positive_weight if target else 1.0
            residual = example_weight * (probability - target)
            curvature = max(example_weight * probability * (1.0 - probability), 1e-8)
            for left in range(dimension + 1):
                gradient[left] += residual * expanded[left]
                for right in range(left, dimension + 1):
                    hessian[left][right] += curvature * expanded[left] * expanded[right]
        for left in range(1, dimension + 1):
            gradient[left] += l2 * coefficients[left]
            hessian[left][left] += l2
        for left in range(dimension + 1):
            hessian[left][left] += 1e-8
            for right in range(left):
                hessian[left][right] = hessian[right][left]

        delta = _solve(hessian, gradient)
        old_loss = _weighted_log_loss(
            x_rows, targets, coefficients, positive_weight, l2
        )
        step = 1.0
        candidate = coefficients
        while step >= 1.0 / 128.0:
            candidate = [value - step * change for value, change in zip(coefficients, delta)]
            if _weighted_log_loss(x_rows, targets, candidate, positive_weight, l2) <= old_loss:
                break
            step /= 2.0
        coefficients = candidate
        if max(abs(step * change) for change in delta) < 1e-6:
            break
    return coefficients


@dataclass
class OrdinalLogisticRouter:
    encoder: FeatureEncoder
    balanced_coefficients: list[float]
    strongest_coefficients: list[float]
    positive_weights: tuple[float, float] = (3.0, 5.0)
    l2: float = 1.0

    @classmethod
    def fit(
        cls,
        feature_rows: Sequence[Mapping[str, float | str]],
        labels: Sequence[int],
        *,
        positive_weights: tuple[float, float] = (3.0, 5.0),
        l2: float = 1.0,
        max_iterations: int = 40,
    ) -> "OrdinalLogisticRouter":
        encoder = FeatureEncoder.fit(feature_rows)
        matrix = encoder.transform(feature_rows)
        balanced_targets = [int(label >= 1) for label in labels]
        strongest_targets = [int(label >= 2) for label in labels]
        return cls(
            encoder=encoder,
            balanced_coefficients=_fit_logistic_head(
                matrix,
                balanced_targets,
                positive_weight=positive_weights[0],
                l2=l2,
                max_iterations=max_iterations,
            ),
            strongest_coefficients=_fit_logistic_head(
                matrix,
                strongest_targets,
                positive_weight=positive_weights[1],
                l2=l2,
                max_iterations=max_iterations,
            ),
            positive_weights=positive_weights,
            l2=l2,
        )

    @property
    def trainable_parameter_count(self) -> int:
        return len(self.balanced_coefficients) + len(self.strongest_coefficients)

    @staticmethod
    def _head_probability(row: Sequence[float], coefficients: Sequence[float]) -> float:
        return _sigmoid(
            coefficients[0]
            + sum(weight * feature for weight, feature in zip(coefficients[1:], row))
        )

    def predict_one(self, features: Mapping[str, float | str]) -> Prediction:
        row = self.encoder.transform_one(features)
        probability_balanced = self._head_probability(row, self.balanced_coefficients)
        probability_strongest = min(
            probability_balanced,
            self._head_probability(row, self.strongest_coefficients),
        )
        score = _score(probability_balanced, probability_strongest)
        return Prediction(score, probability_balanced, probability_strongest, tier_for_score(score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "two_head_ordinal_logistic",
            "encoder": self.encoder.to_dict(),
            "balanced_coefficients": self.balanced_coefficients,
            "strongest_coefficients": self.strongest_coefficients,
            "positive_weights": list(self.positive_weights),
            "l2": self.l2,
            "trainable_parameter_count": self.trainable_parameter_count,
            "score_formula": "10 + 45*p_balanced + 35*min(p_strongest,p_balanced)",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrdinalLogisticRouter":
        return cls(
            encoder=FeatureEncoder.from_dict(payload["encoder"]),
            balanced_coefficients=[float(value) for value in payload["balanced_coefficients"]],
            strongest_coefficients=[float(value) for value in payload["strongest_coefficients"]],
            positive_weights=tuple(float(value) for value in payload["positive_weights"]),  # type: ignore[arg-type]
            l2=float(payload["l2"]),
        )


@dataclass
class TreeNode:
    value: float
    feature: int | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    def predict(self, row: Sequence[float]) -> float:
        if self.feature is None:
            return self.value
        child = self.left if row[self.feature] <= float(self.threshold) else self.right
        if child is None:  # defensive fallback for malformed external artifacts
            return self.value
        return child.predict(row)

    @property
    def leaf_count(self) -> int:
        if self.feature is None:
            return 1
        return (self.left.leaf_count if self.left else 0) + (
            self.right.leaf_count if self.right else 0
        )

    def to_dict(self) -> dict[str, Any]:
        if self.feature is None:
            return {"value": self.value}
        return {
            "value": self.value,
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left.to_dict() if self.left else None,
            "right": self.right.to_dict() if self.right else None,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TreeNode":
        return cls(
            value=float(payload["value"]),
            feature=int(payload["feature"]) if "feature" in payload else None,
            threshold=float(payload["threshold"]) if "threshold" in payload else None,
            left=cls.from_dict(payload["left"]) if payload.get("left") else None,
            right=cls.from_dict(payload["right"]) if payload.get("right") else None,
        )


def _candidate_thresholds(matrix: Sequence[Sequence[float]], max_bins: int) -> list[list[float]]:
    candidates: list[list[float]] = []
    for feature in range(len(matrix[0])):
        values = sorted({row[feature] for row in matrix})
        if len(values) <= 1:
            candidates.append([])
            continue
        midpoints = [(left + right) / 2.0 for left, right in zip(values, values[1:])]
        if len(midpoints) > max_bins:
            chosen = {
                midpoints[min(len(midpoints) - 1, int(index * len(midpoints) / max_bins))]
                for index in range(max_bins)
            }
            midpoints = sorted(chosen)
        candidates.append(midpoints)
    return candidates


def _build_tree(
    matrix: Sequence[Sequence[float]],
    gradients: Sequence[float],
    hessians: Sequence[float],
    indices: list[int],
    candidates: Sequence[Sequence[float]],
    *,
    depth: int,
    max_depth: int,
    min_samples_leaf: int,
    l2: float,
    learning_rate: float,
) -> TreeNode:
    total_gradient = sum(gradients[index] for index in indices)
    total_hessian = sum(hessians[index] for index in indices)
    leaf_value = -learning_rate * total_gradient / (total_hessian + l2)
    if depth >= max_depth or len(indices) < 2 * min_samples_leaf:
        return TreeNode(value=leaf_value)

    parent_term = total_gradient * total_gradient / (total_hessian + l2)
    best: tuple[float, int, float, list[int], list[int]] | None = None
    for feature, thresholds in enumerate(candidates):
        if not thresholds:
            continue
        bucket_count = len(thresholds) + 1
        counts = [0] * bucket_count
        gradient_sums = [0.0] * bucket_count
        hessian_sums = [0.0] * bucket_count
        for index in indices:
            bucket = bisect_right(thresholds, matrix[index][feature])
            counts[bucket] += 1
            gradient_sums[bucket] += gradients[index]
            hessian_sums[bucket] += hessians[index]
        left_count = 0
        left_gradient = 0.0
        left_hessian = 0.0
        for split_index, threshold in enumerate(thresholds):
            left_count += counts[split_index]
            left_gradient += gradient_sums[split_index]
            left_hessian += hessian_sums[split_index]
            if left_count < min_samples_leaf or len(indices) - left_count < min_samples_leaf:
                continue
            right_gradient = total_gradient - left_gradient
            right_hessian = total_hessian - left_hessian
            gain = 0.5 * (
                left_gradient * left_gradient / (left_hessian + l2)
                + right_gradient * right_gradient / (right_hessian + l2)
                - parent_term
            )
            candidate = (gain, feature, threshold, [], [])
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None or best[0] <= 1e-9:
        return TreeNode(value=leaf_value)
    _, feature, threshold, _, _ = best
    left_indices = [index for index in indices if matrix[index][feature] <= threshold]
    right_indices = [index for index in indices if matrix[index][feature] > threshold]
    return TreeNode(
        value=leaf_value,
        feature=feature,
        threshold=threshold,
        left=_build_tree(
            matrix,
            gradients,
            hessians,
            left_indices,
            candidates,
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2=l2,
            learning_rate=learning_rate,
        ),
        right=_build_tree(
            matrix,
            gradients,
            hessians,
            right_indices,
            candidates,
            depth=depth + 1,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2=l2,
            learning_rate=learning_rate,
        ),
    )


def _fit_boosted_head(
    matrix: Sequence[Sequence[float]],
    targets: Sequence[int],
    *,
    positive_weight: float,
    tree_count: int,
    max_depth: int,
    min_samples_leaf: int,
    learning_rate: float,
    l2: float,
    max_bins: int,
) -> tuple[float, list[TreeNode]]:
    weighted_positives = positive_weight * sum(targets)
    weighted_total = weighted_positives + len(targets) - sum(targets)
    base_value = _logit(weighted_positives / weighted_total)
    logits = [base_value] * len(targets)
    trees: list[TreeNode] = []
    candidates = _candidate_thresholds(matrix, max_bins)
    all_indices = list(range(len(matrix)))
    for _ in range(tree_count):
        probabilities = [_sigmoid(value) for value in logits]
        weights = [positive_weight if target else 1.0 for target in targets]
        gradients = [
            weight * (probability - target)
            for weight, probability, target in zip(weights, probabilities, targets)
        ]
        hessians = [
            max(weight * probability * (1.0 - probability), 1e-8)
            for weight, probability in zip(weights, probabilities)
        ]
        tree = _build_tree(
            matrix,
            gradients,
            hessians,
            all_indices,
            candidates,
            depth=0,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2=l2,
            learning_rate=learning_rate,
        )
        trees.append(tree)
        logits = [value + tree.predict(row) for value, row in zip(logits, matrix)]
    return base_value, trees


@dataclass
class OrdinalBoostedRouter:
    encoder: FeatureEncoder
    balanced_base: float
    balanced_trees: list[TreeNode]
    strongest_base: float
    strongest_trees: list[TreeNode]
    hyperparameters: dict[str, float | int]
    positive_weights: tuple[float, float] = (3.0, 5.0)

    @classmethod
    def fit(
        cls,
        feature_rows: Sequence[Mapping[str, float | str]],
        labels: Sequence[int],
        *,
        positive_weights: tuple[float, float] = (3.0, 5.0),
        tree_count: int = 60,
        max_depth: int = 2,
        min_samples_leaf: int = 25,
        learning_rate: float = 0.05,
        l2: float = 1.0,
        max_bins: int = 16,
    ) -> "OrdinalBoostedRouter":
        encoder = FeatureEncoder.fit(feature_rows)
        matrix = encoder.transform(feature_rows)
        common = {
            "tree_count": tree_count,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "learning_rate": learning_rate,
            "l2": l2,
            "max_bins": max_bins,
        }
        balanced_base, balanced_trees = _fit_boosted_head(
            matrix,
            [int(label >= 1) for label in labels],
            positive_weight=positive_weights[0],
            **common,
        )
        strongest_base, strongest_trees = _fit_boosted_head(
            matrix,
            [int(label >= 2) for label in labels],
            positive_weight=positive_weights[1],
            **common,
        )
        return cls(
            encoder=encoder,
            balanced_base=balanced_base,
            balanced_trees=balanced_trees,
            strongest_base=strongest_base,
            strongest_trees=strongest_trees,
            hyperparameters=common,
            positive_weights=positive_weights,
        )

    @property
    def learned_leaf_count(self) -> int:
        return sum(tree.leaf_count for tree in self.balanced_trees + self.strongest_trees)

    @staticmethod
    def _head_probability(row: Sequence[float], base: float, trees: Iterable[TreeNode]) -> float:
        return _sigmoid(base + sum(tree.predict(row) for tree in trees))

    def predict_one(self, features: Mapping[str, float | str]) -> Prediction:
        row = self.encoder.transform_one(features)
        probability_balanced = self._head_probability(
            row, self.balanced_base, self.balanced_trees
        )
        probability_strongest = min(
            probability_balanced,
            self._head_probability(row, self.strongest_base, self.strongest_trees),
        )
        score = _score(probability_balanced, probability_strongest)
        return Prediction(score, probability_balanced, probability_strongest, tier_for_score(score))

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_type": "two_head_ordinal_gradient_boosting",
            "encoder": self.encoder.to_dict(),
            "balanced_base": self.balanced_base,
            "balanced_trees": [tree.to_dict() for tree in self.balanced_trees],
            "strongest_base": self.strongest_base,
            "strongest_trees": [tree.to_dict() for tree in self.strongest_trees],
            "hyperparameters": self.hyperparameters,
            "positive_weights": list(self.positive_weights),
            "learned_leaf_count": self.learned_leaf_count,
            "score_formula": "10 + 45*p_balanced + 35*min(p_strongest,p_balanced)",
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OrdinalBoostedRouter":
        return cls(
            encoder=FeatureEncoder.from_dict(payload["encoder"]),
            balanced_base=float(payload["balanced_base"]),
            balanced_trees=[TreeNode.from_dict(tree) for tree in payload["balanced_trees"]],
            strongest_base=float(payload["strongest_base"]),
            strongest_trees=[TreeNode.from_dict(tree) for tree in payload["strongest_trees"]],
            hyperparameters=dict(payload["hyperparameters"]),
            positive_weights=tuple(float(value) for value in payload["positive_weights"]),  # type: ignore[arg-type]
        )

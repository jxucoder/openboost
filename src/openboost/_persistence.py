"""Model persistence utilities for OpenBoost.

Phase 20.1: Model Persistence

Provides save/load functionality for all OpenBoost models using joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from ._core._growth import TreeStructure

T = TypeVar("T", bound="PersistenceMixin")


def _to_numpy(arr: Any) -> np.ndarray | None:
    """Convert array to numpy, handling GPU arrays.

    Args:
        arr: Array (numpy, cuda device array, or None)

    Returns:
        numpy array or None
    """
    if arr is None:
        return None

    # Handle numba cuda arrays
    if hasattr(arr, "copy_to_host"):
        return arr.copy_to_host()

    # Handle cupy arrays
    if hasattr(arr, "get"):
        return arr.get()

    # Already numpy or compatible
    return np.asarray(arr)


def _tree_to_dict(tree: TreeStructure) -> dict[str, Any]:
    """Convert TreeStructure to a serializable dictionary.

    Args:
        tree: TreeStructure instance

    Returns:
        Dictionary with tree data
    """
    from ._core._growth import ScalarLeaves, VectorLeaves

    data = {
        "features": _to_numpy(tree.features),
        "thresholds": _to_numpy(tree.thresholds),
        "left_children": _to_numpy(tree.left_children),
        "right_children": _to_numpy(tree.right_children),
        "n_nodes": tree.n_nodes,
        "depth": tree.depth,
        "n_features": tree.n_features,
        "is_symmetric": tree.is_symmetric,
    }

    # Handle leaf values (can be array or LeafValues subclass)
    # Note: Check specific subclasses first since numpy arrays match the
    # LeafValues Protocol (it's runtime_checkable)
    if isinstance(tree.values, ScalarLeaves):
        data["values"] = _to_numpy(tree.values.values)
        data["values_type"] = "scalar"
    elif isinstance(tree.values, VectorLeaves):
        data["values"] = _to_numpy(tree.values.values)
        data["values_type"] = "vector"
    else:
        # Regular numpy array (or compatible)
        data["values"] = _to_numpy(tree.values)
        data["values_type"] = "array"

    # Symmetric tree data
    if tree.is_symmetric:
        data["level_features"] = _to_numpy(tree.level_features)
        data["level_thresholds"] = _to_numpy(tree.level_thresholds)

    # Phase 14: Missing value handling
    if hasattr(tree, "missing_go_left") and tree.missing_go_left is not None:
        data["missing_go_left"] = _to_numpy(tree.missing_go_left)

    # Phase 14.3: Categorical support
    if hasattr(tree, "is_categorical") and tree.is_categorical is not None:
        data["is_categorical"] = _to_numpy(tree.is_categorical)
    if hasattr(tree, "category_masks") and tree.category_masks is not None:
        data["category_masks"] = _to_numpy(tree.category_masks)

    return data


def _linear_leaf_tree_to_dict(tree) -> dict[str, Any]:
    """Convert LinearLeafTree to a serializable dictionary.

    Args:
        tree: LinearLeafTree instance

    Returns:
        Dictionary with tree data
    """
    return {
        "tree_structure": _tree_to_dict(tree.tree_structure),
        "leaf_weights": _to_numpy(tree.leaf_weights),
        "leaf_features": tree.leaf_features,
        "leaf_ids": tree.leaf_ids,
        "n_features": tree.n_features,
        "_type": "LinearLeafTree",
    }


def _dict_to_linear_leaf_tree(data: dict[str, Any]):
    """Reconstruct LinearLeafTree from dictionary.

    Args:
        data: Dictionary with tree data

    Returns:
        LinearLeafTree instance
    """
    from ._models._linear_leaf import LinearLeafTree

    return LinearLeafTree(
        tree_structure=_dict_to_tree(data["tree_structure"]),
        leaf_weights=data["leaf_weights"],
        leaf_features=data["leaf_features"],
        leaf_ids=data["leaf_ids"],
        n_features=data["n_features"],
    )


def _dict_to_tree(data: dict[str, Any]) -> TreeStructure:
    """Reconstruct TreeStructure from dictionary.

    Args:
        data: Dictionary with tree data

    Returns:
        TreeStructure instance
    """
    from ._core._growth import TreeStructure, ScalarLeaves, VectorLeaves

    # Handle leaf values based on type
    values_type = data.get("values_type", "array")
    values_arr = data["values"]

    if values_type == "scalar":
        values = ScalarLeaves(values_arr)
    elif values_type == "vector":
        values = VectorLeaves(values_arr)
    else:
        values = values_arr

    tree = TreeStructure(
        features=data["features"],
        thresholds=data["thresholds"],
        left_children=data["left_children"],
        right_children=data["right_children"],
        values=values,
        n_nodes=data["n_nodes"],
        depth=data["depth"],
        n_features=data["n_features"],
        is_symmetric=data.get("is_symmetric", False),
        level_features=data.get("level_features"),
        level_thresholds=data.get("level_thresholds"),
    )

    # Phase 14: Missing value handling
    if "missing_go_left" in data:
        tree.missing_go_left = data["missing_go_left"]

    # Phase 14.3: Categorical support
    if "is_categorical" in data:
        tree.is_categorical = data["is_categorical"]
    if "category_masks" in data:
        tree.category_masks = data["category_masks"]

    return tree


class PersistenceMixin:
    """Mixin class providing save/load functionality for models.

    Usage:
        @dataclass
        class MyModel(PersistenceMixin):
            n_trees: int = 100
            trees_: list = field(default_factory=list, init=False)

            def _get_persist_attrs(self) -> list[str]:
                return ['n_trees', 'trees_']
    """

    def _get_persist_attrs(self) -> list[str]:
        """Return list of attribute names to persist.

        Override in subclass to customize what gets saved.
        By default, saves all non-private attributes.

        Returns:
            List of attribute names
        """
        # Get all attributes from dataclass fields
        if hasattr(self, "__dataclass_fields__"):
            return list(self.__dataclass_fields__.keys())
        # Fallback: all non-private attributes
        return [k for k in vars(self).keys() if not k.startswith("_")]

    def _to_state_dict(self) -> dict[str, Any]:
        """Convert model to a serializable state dictionary.

        Returns:
            Dictionary containing all model state
        """
        state = {"__class__": type(self).__name__}

        for attr in self._get_persist_attrs():
            value = getattr(self, attr, None)

            # Handle tree dict (distributional models: param_name -> tree list)
            if attr == "trees_" and isinstance(value, dict):
                state[attr] = {
                    k: [_tree_to_dict(t) for t in v] for k, v in value.items()
                }
                state["_trees_type"] = "dict"
            # Handle nested tree lists (multiclass: list of tree lists)
            elif attr == "trees_" and isinstance(value, list) and value and isinstance(value[0], list):
                state[attr] = [[_tree_to_dict(t) for t in trees] for trees in value]
                state["_trees_type"] = "nested_list"
            # Handle LinearLeafTree lists (check for tree_structure attribute)
            elif attr == "trees_" and isinstance(value, list) and value and hasattr(value[0], "tree_structure"):
                state[attr] = [_linear_leaf_tree_to_dict(t) for t in value]
                state["_trees_type"] = "linear_leaf"
            # Handle simple tree lists
            elif attr == "trees_" and isinstance(value, list):
                state[attr] = [_tree_to_dict(t) for t in value]
                state["_trees_type"] = "list"
            # Handle BinnedArray (save bin edges for transform)
            elif attr == "X_binned_":
                if value is not None:
                    state["_bin_edges"] = value.bin_edges
                    state["_n_features"] = value.n_features
                    if hasattr(value, "has_missing"):
                        state["_has_missing"] = _to_numpy(value.has_missing)
                    if hasattr(value, "is_categorical"):
                        state["_is_categorical"] = _to_numpy(value.is_categorical)
                    if hasattr(value, "category_maps"):
                        state["_category_maps"] = value.category_maps
                    if hasattr(value, "n_categories"):
                        state["_n_categories"] = _to_numpy(value.n_categories)
                continue  # Don't save the full BinnedArray
            # Handle loss function (save name, not function)
            elif attr == "_loss_fn":
                continue  # Skip - will be recreated from loss param
            # Skip distribution instance (will be recreated from distribution param)
            elif attr == "distribution_":
                continue
            # Handle arrays
            elif hasattr(value, "shape"):
                state[attr] = _to_numpy(value)
            # Everything else
            else:
                state[attr] = value

        return state

    def _from_state_dict(self, state: dict[str, Any]) -> None:
        """Restore model state from dictionary.

        Args:
            state: Dictionary containing model state
        """
        trees_type = state.get("_trees_type", "list")

        for attr, value in state.items():
            if attr in ("__class__", "_trees_type"):
                continue

            # Handle tree structures based on stored type
            if attr == "trees_":
                if trees_type == "dict" and isinstance(value, dict):
                    # Distributional: param_name -> tree list
                    setattr(
                        self,
                        attr,
                        {k: [_dict_to_tree(d) for d in v] for k, v in value.items()},
                    )
                elif trees_type == "nested_list" and isinstance(value, list):
                    # Multi-class: list of list of tree dicts
                    setattr(
                        self,
                        attr,
                        [[_dict_to_tree(d) for d in trees] for trees in value],
                    )
                elif trees_type == "linear_leaf" and isinstance(value, list):
                    # LinearLeafTree list
                    setattr(
                        self,
                        attr,
                        [_dict_to_linear_leaf_tree(d) for d in value],
                    )
                elif isinstance(value, list) and value and isinstance(value[0], dict):
                    # Single-output: list of tree dicts
                    setattr(self, attr, [_dict_to_tree(d) for d in value])
                else:
                    setattr(self, attr, value)
            else:
                setattr(self, attr, value)

        # Restore bin edges for transform
        if "_bin_edges" in state:
            from ._array import BinnedArray
            import numpy as np
            
            # Create a minimal BinnedArray with just bin edges for transform
            n_features = state.get("_n_features", len(state["_bin_edges"]))
            has_missing = state.get("_has_missing", np.array([], dtype=np.bool_))
            is_categorical = state.get("_is_categorical", np.array([], dtype=np.bool_))
            category_maps = state.get("_category_maps", [])
            n_categories = state.get("_n_categories", np.array([], dtype=np.int32))
            
            # Create placeholder data (empty, just need structure for transform)
            placeholder_data = np.zeros((n_features, 0), dtype=np.uint8)
            
            self.X_binned_ = BinnedArray(
                data=placeholder_data,
                bin_edges=state["_bin_edges"],
                n_features=n_features,
                n_samples=0,  # Placeholder
                device="cpu",
                has_missing=has_missing if isinstance(has_missing, np.ndarray) else np.array(has_missing, dtype=np.bool_),
                is_categorical=is_categorical if isinstance(is_categorical, np.ndarray) else np.array(is_categorical, dtype=np.bool_),
                category_maps=category_maps,
                n_categories=n_categories if isinstance(n_categories, np.ndarray) else np.array(n_categories, dtype=np.int32),
            )
        
        # Call post-load hook if defined (for recreating derived attributes)
        if hasattr(self, "_post_load"):
            self._post_load()

    def save(self, path: str | Path) -> None:
        """Save model to file.

        Uses joblib for efficient serialization of numpy arrays.

        Args:
            path: File path. Recommended extensions: .joblib, .pkl

        Example:
            >>> model = ob.GradientBoosting(n_trees=100)
            >>> model.fit(X_train, y_train)
            >>> model.save('my_model.joblib')
        """
        import joblib

        path = Path(path)
        state = self._to_state_dict()
        joblib.dump(state, path)

    @classmethod
    def load(cls: type[T], path: str | Path) -> T:
        """Load model from file.

        Args:
            path: File path to load from

        Returns:
            Loaded model instance

        Example:
            >>> model = ob.GradientBoosting.load('my_model.joblib')
            >>> predictions = model.predict(X_test)
        """
        import joblib

        path = Path(path)
        state = joblib.load(path)

        # Verify class matches
        saved_class = state.get("__class__", "")
        if saved_class and saved_class != cls.__name__:
            raise ValueError(
                f"Model was saved as {saved_class}, but loading as {cls.__name__}. "
                f"Use {saved_class}.load() instead."
            )

        # Create instance without calling __init__
        model = cls.__new__(cls)

        # Initialize default values from dataclass
        if hasattr(cls, "__dataclass_fields__"):
            from dataclasses import MISSING

            for name, field_info in cls.__dataclass_fields__.items():
                # Check for default value (not MISSING)
                if field_info.default is not MISSING:
                    setattr(model, name, field_info.default)
                # Check for default_factory (not MISSING)
                elif field_info.default_factory is not MISSING:
                    setattr(model, name, field_info.default_factory())
                # No default - leave unset, will be set by _from_state_dict

        # Restore state
        model._from_state_dict(state)

        return model

    def __getstate__(self) -> dict[str, Any]:
        """Support for pickle serialization."""
        return self._to_state_dict()

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Support for pickle deserialization."""
        self._from_state_dict(state)

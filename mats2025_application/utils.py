"""Data generating code from epsilon-transformers"""
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Iterator, cast
from jaxtyping import Float
import numpy as np
from scipy.stats import entropy


@dataclass
class ProcessHistory:
    symbols: list[int]
    states: list[str]

    def __post_init__(self):
        assert len(self.symbols) == len(
            self.states
        ), "length of symbols & states must be the same"

    def __len__(self):
        return len(self.states)
    
class MixedStateTreeNode:
    state_prob_vector: Float[np.ndarray, "n_states"]
    path: list[int]
    children: set["MixedStateTreeNode"]
    emission_prob: float

    def __init__(
        self,
        state_prob_vector: Float[np.ndarray, "n_states"],
        children: set["MixedStateTreeNode"],
        path: list[int],
        emission_prob: float,
    ):
        self.state_prob_vector = state_prob_vector
        self.path = path
        self.children = children
        self.emission_prob = emission_prob

    def add_child(self, child: "MixedStateTreeNode"):
        self.children.add(child)


class MixedStateTree:
    root_node: MixedStateTreeNode
    process: str
    depth: int
    nodes: set[MixedStateTreeNode]

    @property
    def belief_states(self) -> list[Float[np.ndarray, "num_states"]]:
        return [x.state_prob_vector for x in self.nodes]

    @property
    def paths(self) -> list[list[int]]:
        return [x.path for x in self.nodes]

    @property
    def paths_and_belief_states(
        self,
    ) -> tuple[list[list[int]], list[Float[np.ndarray, "n_states"]]]:
        return self.paths, self.belief_states

    @property
    def block_entropy(self) -> Float[np.ndarray, "depth"]:
        depth_emission_probs = self._traverse(
            node=self.root_node, depth=0, accumulated_prob=1.0
        )
        block_entropy = np.array(
            [entropy(probs) if probs else 0 for probs in depth_emission_probs]
        )
        return block_entropy

    @property
    def myopic_entropy(self) -> Float[np.ndarray, "depth-1"]:
        return np.diff(self.block_entropy)

    def __init__(
        self,
        root_node: MixedStateTreeNode,
        process: str,
        nodes: set[MixedStateTreeNode],
        depth: int,
    ):
        self.root_node = root_node
        self.process = process
        self.nodes = nodes
        self.depth = depth

    def _traverse(
        self, node: MixedStateTreeNode, depth: int, accumulated_prob: float
    ) -> list[list[float]]:
        stack = deque([(node, depth, accumulated_prob)])
        depth_emission_probs: list[list[float]] = [[] for _ in range(self.depth)]

        while stack:
            node, depth, accumulated_prob = stack.pop()
            if depth < self.depth:
                if node is not self.root_node:
                    depth_emission_probs[depth].append(
                        accumulated_prob * node.emission_prob
                    )
                for child in node.children:
                    stack.append(
                        (
                            child,
                            depth + 1,
                            (
                                accumulated_prob * node.emission_prob
                                if node is not self.root_node
                                else 1.0
                            ),
                        )
                    )

        return depth_emission_probs

    def path_to_beliefs(
        self, path: list[int]
    ) -> Float[np.ndarray, "path_length n_states"]:
        assert (
            len(path) <= self.depth
        ), f"path length: {len(path)} is too long . Tree has depth of {self.depth}"

        belief_states = []
        current_node = self.root_node
        for i in range(len(path)):
            sub_path = path[: i + 1]
            for child in current_node.children:
                if child.path == sub_path:
                    belief_states.append(child.state_prob_vector)
                    current_node = child
                    break

        assert current_node.path == path, f"{path} is not a valid path for this process"
        assert len(belief_states) == len(path)
        return np.stack(belief_states)

    def build_msp_transition_matrix(
        self,
    ) -> Float[np.ndarray, "num_emission num_msp_nodes num_msp_nodes"]:
        seen_prob_vectors = {}
        max_state_index = (
            -1
        )  # To keep track of the last index assigned to a unique state
        queue = deque(
            [
                (self.root_node, cast(int | None, None), -1, 0.0)
            ]  # cast None as a value that could be an int
        )  # (node, emitted_symbol, parent_state_index, emission_prob)
        # get the number of symbols by looking at all entries of all paths and finding the max index
        num_symbols = len(np.unique(np.concatenate([np.unique(x) for x in self.paths])))
        num_nodes = len(self.nodes)

        # Assuming we know the number of symbols and maximum states to expect
        M = np.zeros((num_symbols, num_nodes, num_nodes))  # Adjust size appropriately

        while queue:
            current_node, emitted_symbol, from_state_index, emission_prob = (
                queue.popleft()
            )
            rounded_vector = np.around(current_node.state_prob_vector, decimals=5)
            vector_tuple = tuple(rounded_vector.tolist())

            # Check if we've seen this state vector before
            if vector_tuple not in seen_prob_vectors:
                max_state_index += 1
                seen_prob_vectors[vector_tuple] = max_state_index
                to_state_index = max_state_index
            else:
                to_state_index = seen_prob_vectors[vector_tuple]

            # Only add this if there's a valid symbol and from_state_index
            if emitted_symbol is not None and from_state_index != -1:
                M[emitted_symbol, from_state_index, to_state_index] = emission_prob

            # Check and add children to the queue
            for child in current_node.children:
                if child.path:
                    child_symbol = child.path[
                        -1
                    ]  # Assume last element of the path is the symbol
                    child_emission_prob = child.emission_prob
                    queue.append(
                        (child, child_symbol, to_state_index, child_emission_prob)
                    )

        # delete entries that were never visited
        M = M[:, : max_state_index + 1, : max_state_index + 1]

        return M

    def _get_nodes_at_depth(self, depth: int) -> set[MixedStateTreeNode]:
        return {n for n in self.nodes if len(n.path) == depth}


class Process(ABC):
    name: str
    transition_matrix: Float[np.ndarray, "vocab_len num_states num_states"]
    state_names_dict: dict[str, int]
    vocab_len: int
    num_states: int

    @property
    def steady_state_vector(self) -> Float[np.ndarray, "num_states"]:
        state_transition_matrix = np.sum(self.transition_matrix, axis=0)

        eigenvalues, eigenvectors = np.linalg.eig(state_transition_matrix.T)
        steady_state_vector = eigenvectors[:, np.isclose(eigenvalues, 1)].real
        normalized_steady_state_vector = steady_state_vector / steady_state_vector.sum()
        out: np.ndarray = normalized_steady_state_vector[:, 0]

        assert out.ndim == 1
        assert len(out) == self.num_states
        return out

    @property
    def is_unifilar(self) -> bool:
        # For each state, check if there are multiple transitions for each symbol
        for i in range(self.num_states):
            for j in range(self.vocab_len):
                # If there are multiple transitions, return False
                if np.count_nonzero(self.transition_matrix[j, i, :]) > 1:
                    return False
        return True

    def __init__(self):
        self.transition_matrix, self.state_names_dict = self._create_hmm()

        if (
            len(self.transition_matrix.shape) != 3
            or self.transition_matrix.shape[1] != self.transition_matrix.shape[2]
        ):
            raise ValueError(
                "Transition matrix should have 3 axes and the final two dims shoulds be square"
            )

        if self.transition_matrix.shape[1] != self.transition_matrix.shape[2]:
            raise ValueError("Transition matrix should be square")

        transition = self.transition_matrix.sum(axis=0)
        if not np.allclose(transition.sum(axis=1), 1.0):
            raise ValueError("Transition matrix should be stochastic and sum to 1")

        self.vocab_len = self.transition_matrix.shape[0]
        self.num_states = self.transition_matrix.shape[1]

    @abstractmethod
    def _create_hmm(
        self,
    ) -> tuple[Float[np.ndarray, "vocab_len num_states num_states"], dict[str, int]]:
        """
        Create the HMM which defines the process.

        Returns:
        numpy.ndarray: The transition tensor for the epsilon machine.
        dict: A dictionary mapping state names to indices.
        """
        ...

    def __str__(self):
        return (
            f"{self.name} Process\n"
            f"Number of states: {self.num_states}\n"
            f"Vocabulary length: {self.vocab_len}\n"
            f"Transition matrix shape: {self.transition_matrix.shape}"
        )

    def _sample_emission(self, current_state_idx: int | None = None) -> int:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )

        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"

        p = self.transition_matrix[:, current_state_idx, :].sum(axis=1)
        emission = np.random.choice(self.vocab_len, p=p)
        return emission

    def yield_emissions(
        self, sequence_len: int, current_state_idx: int | None = None
    ) -> Iterator[int]:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )
        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"
        for _ in range(sequence_len):
            emission, next_state_idx = self._sample_emission_and_next_state(
                current_state_idx
            )
            yield emission
            current_state_idx = next_state_idx

    def _sample_emission_and_next_state(
        self, current_state_idx: int
    ) -> tuple[int, int]:
        transition_probs = self.transition_matrix[:, current_state_idx, :] # (vocab_len, state)
        emission_next_state_idx = np.random.choice(
            transition_probs.size, p=transition_probs.ravel()
        )
        emission = emission_next_state_idx // self.num_states
        next_state_idx = emission_next_state_idx % self.num_states
        return emission, next_state_idx

    def yield_emission_histories(
        self, sequence_len: int, num_sequences: int
    ) -> Iterator[list[int]]:
        for _ in range(num_sequences):
            yield [x for x in self.yield_emissions(sequence_len=sequence_len)]

    def generate_process_history(
        self, total_length: int, current_state_idx: int | None = None
    ) -> ProcessHistory:
        if current_state_idx is None:
            current_state_idx = np.random.choice(
                self.num_states, p=self.steady_state_vector
            )
        assert (
            0 <= current_state_idx < self.num_states
        ), "current_state_index must be positive & less than num_states"

        index_to_state_names_dict = {v: k for k, v in self.state_names_dict.items()}

        symbols = []
        states = []

        for _ in range(total_length):
            states.append(index_to_state_names_dict[current_state_idx])
            emission, next_state_idx = self._sample_emission_and_next_state(
                current_state_idx
            )
            symbols.append(emission)
            current_state_idx = next_state_idx

        return ProcessHistory(symbols=symbols, states=states)

    # TODO: You can get rid of the stack, and just iterate through the nodes & the depth as tuples
    def derive_mixed_state_presentation(self, depth: int) -> MixedStateTree:
        tree_root = MixedStateTreeNode(
            state_prob_vector=self.steady_state_vector,
            children=set(),
            path=[],
            emission_prob=0,
        )
        nodes = set([tree_root])

        stack: deque[
            tuple[MixedStateTreeNode, Float[np.ndarray, "num_states"], list[int], int]
        ] = deque([(tree_root, self.steady_state_vector, [], 0)])
        while stack:
            current_node, state_prob_vector, current_path, current_depth = stack.pop()
            if current_depth < depth:
                emission_probs = _compute_emission_probabilities(
                    self, state_prob_vector
                )
                for emission in range(self.vocab_len):
                    if emission_probs[emission] > 0:
                        next_state_prob_vector = _compute_next_distribution(
                            self.transition_matrix, state_prob_vector, emission
                        )
                        child_path = current_path + [emission]
                        child_node = MixedStateTreeNode(
                            state_prob_vector=next_state_prob_vector,
                            path=child_path,
                            children=set(),
                            emission_prob=emission_probs[emission],
                        )
                        current_node.add_child(child_node)

                        stack.append(
                            (
                                child_node,
                                next_state_prob_vector,
                                child_path,
                                current_depth + 1,
                            )
                        )
            nodes.add(current_node)

        return MixedStateTree(
            root_node=tree_root, process=self.name, nodes=nodes, depth=depth
        )


def _compute_emission_probabilities(
    hmm: Process, state_prob_vector: Float[np.ndarray, "num_states"]
) -> Float[np.ndarray, "vocab_len"]:
    """
    Compute the probabilities associated with each emission given the current mixed state.
    """
    T = hmm.transition_matrix
    emission_probs = np.einsum("s,esd->ed", state_prob_vector, T).sum(axis=1)
    emission_probs /= emission_probs.sum()
    return emission_probs


def _compute_next_distribution(
    epsilon_machine: Float[np.ndarray, "vocab_len num_states num_states"],
    current_state_prob_vector: Float[np.ndarray, "num_states"],
    current_emission: int,
) -> Float[np.ndarray, "num_states"]:
    """
    Compute the next mixed state distribution for a given output.
    """
    X_next = np.einsum(
        "sd, s -> d", epsilon_machine[current_emission], current_state_prob_vector
    )
    return X_next / np.sum(X_next) if np.sum(X_next) != 0 else X_next


class Mess3(Process):
    def __init__(self, x=0.15, a=0.6):
        self.name = "mess3"
        self.x = x
        self.a = a
        super().__init__()

    def _create_hmm(self):
        T = np.zeros((3, 3, 3))
        state_names = {"A": 0, "B": 1, "C": 2}
        b = (1 - self.a) / 2
        y = 1 - 2 * self.x

        ay = self.a * y
        bx = b * self.x
        by = b * y
        ax = self.a * self.x

        T[0, :, :] = [[ay, bx, bx], [ax, by, bx], [ax, bx, by]]
        T[1, :, :] = [[by, ax, bx], [bx, ay, bx], [bx, ax, by]]
        T[2, :, :] = [[by, bx, ax], [bx, by, ax], [bx, bx, ay]]

        return T, state_names
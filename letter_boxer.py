import argparse
import string
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import chain, pairwise
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, List, Optional, Set, Tuple

from datrie import Trie
from tqdm import tqdm


def build_trie_from_file(filename: Path):
    """
    Build a trie from the dictionary file
    """

    trie = Trie(string.ascii_lowercase)

    with open(filename, "r") as f:
        for line in f:
            word = line.strip()
            if word.isalpha() and word.islower() and len(word) > 2:
                trie[word] = True

    return trie


class SolutionCandidate:
    char_set: ClassVar[FrozenSet[str]] = frozenset()
    word_list: List[str]
    residue_size_history: List[int]
    residue: FrozenSet[str]

    def __init__(
        self,
        word_list: Optional[List[str]] = None,
        residue_size_history: Optional[List[int]] = None,
        residue: Optional[FrozenSet[str]] = None,
    ) -> None:
        self.word_list = word_list if word_list is not None else []

        if residue_size_history is not None and residue is not None:
            self.residue_size_history = residue_size_history
            self.residue = residue
        else:
            self.residue_size_history = []
            residue = set(self.char_set)
            for word in self.word_list:
                residue = residue - {c for c in word}
                self.residue_size_history.append(len(residue))
            self.residue = frozenset(residue)

    def add_word(self, word: str) -> "SolutionCandidate":
        next_residue = self.residue - set(word)
        return SolutionCandidate(
            self.word_list + [word],
            self.residue_size_history + [len(next_residue)],
            next_residue,
        )

    def viable_next_word(self, word: str) -> bool:
        next_residue = self.residue - set(word)

        # check if the residue is decreasing
        if len(next_residue) < len(self.residue):
            return True

        # otherwise see if we've been at an equivalent state (same residue size with same ending char)
        for res_size, prior_word in zip(self.residue_size_history, self.word_list):
            if res_size == len(next_residue) and prior_word[-1] == word[-1]:
                return False

        # It may be possible to make progress from here
        return True

    def gamestate_class(self) -> Tuple[Tuple[FrozenSet[str], str], int]:
        return (self.residue, self.word_list[-1][-1]), len(self.word_list)


@dataclass(order=True)
class PrioritizedSolutionCandidate:
    priority: Tuple[int, int]
    candidate: SolutionCandidate = field(compare=False)

    def __init__(self, candidate: SolutionCandidate) -> None:
        self.candidate = candidate
        self.priority = (len(candidate.residue), len(candidate.word_list))


class LetterBoxer:
    sides: Tuple[str, str, str, str]
    charset: FrozenSet[str]
    char_map: Dict[str, FrozenSet[str]]
    trie: Trie
    possible_words: Set[str]
    word_map: Dict[str, Set[str]]

    def __init__(self, sides: Tuple[str, str, str, str], trie: Trie) -> None:
        self.sides = sides
        self.charset = frozenset(chain(*sides))
        self.trie = trie

        self.char_map = {}
        for side in sides:
            for c in side:
                self.char_map[c] = frozenset(set(self.charset) - set(side))

        SolutionCandidate.char_set = self.charset

        self._find_possible_words()
        self._create_word_mapping()

    def _find_possible_words(self) -> None:
        """
        Find all possible words that can be formed from the sides.
        """
        self.possible_words = set()
        for start_char in self.charset:
            for word in self.trie.keys(start_char):
                possible = True
                for c0, c1 in pairwise(word):
                    if c0 not in self.char_map or c1 not in self.char_map[c0]:
                        possible = False
                        break
                if possible:
                    self.possible_words.add(word)
        print(f"Found {len(self.possible_words)} possible words.")

    def _create_word_mapping(self) -> None:
        """
        Create a mapping of characters to possible words that start with that character.
        """
        self.word_map = {}
        for word in self.possible_words:
            if word[0] not in self.word_map:
                self.word_map[word[0]] = set()
            self.word_map[word[0]].add(word)

    def check_solvable(self) -> bool:
        """
        Check if the puzzle is solvable.
        """
        if set(self.charset) - {c for word in self.possible_words for c in word}:
            return False

        # find connected components; if there is a connected component that contains all the characters, the puzzle is solvable
        char_graph = {}
        for char in self.charset:
            char_graph[char] = set()
            if char not in self.word_map:
                continue
            for word in self.word_map[char]:
                char_graph[char] = char_graph[char].union(set(word))

        for start in char_graph:
            visited = set()
            stack = [start]
            while stack:
                curr = stack.pop()
                visited.add(curr)
                stack.extend(char_graph[curr] - visited)
            if visited == self.charset:
                return True

        return False

    def solve(self):
        """
        Solve the puzzle with branch and bound.
        """
        solutions = []
        best = None
        pq = []
        candidates_considered = 0

        seen_states = {}

        if not self.check_solvable():
            print("Puzzle is unsolvable.")
            return solutions

        for word in self.possible_words:
            heappush(pq, PrioritizedSolutionCandidate(SolutionCandidate([word])))

        progress_bar = tqdm(total=1, desc="Processing nodes", dynamic_ncols=True)

        while pq:
            curr = heappop(pq)
            candidates_considered += 1

            gamestate, size = curr.candidate.gamestate_class()
            if (
                gamestate in seen_states and seen_states[gamestate] <= size
            ):  # we'be been here or a more promising state
                continue

            seen_states[gamestate] = size

            # Update the progress bar based on the ratio of candidates considered to total candidates
            progress = candidates_considered / (candidates_considered + len(pq))
            progress_bar.update(progress - progress_bar.n)

            if curr.candidate.residue == frozenset():
                if best is None or len(curr.candidate.word_list) < len(best.word_list):
                    best = curr.candidate
                    pq = list(
                        filter(
                            lambda x: len(x.candidate.word_list) <= len(best.word_list),
                            pq,
                        )
                    )
                    solutions = []
                if best is not None and len(curr.candidate.word_list) == len(
                    best.word_list
                ):
                    solutions.append(curr.candidate.word_list)
                continue

            if best is not None and len(curr.candidate.word_list) >= len(
                best.word_list
            ):
                continue

            last_char = curr.candidate.word_list[-1][
                -1
            ]  # all candidates have at least one word
            possible_next_words = (
                self.word_map[last_char] if last_char in self.word_map else []
            )
            for next_word in possible_next_words:
                heappush(
                    pq,
                    PrioritizedSolutionCandidate(curr.candidate.add_word(next_word)),
                )

        return solutions


def generate_random_letter_set() -> Tuple[str, str, str, str]:
    """
    Generate a random set of letters for the puzzle.
    """
    import random

    charset = string.ascii_lowercase
    chars = random.sample(charset, k=12)
    return tuple("".join(chars[i : i + 3]) for i in range(0, 12, 3))


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal solutions to the Letter Boxer puzzle.",
        usage="%(prog)s --dictionary <path> [--random | s0 s1 s2 s3]",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default="/usr/share/dict/words",
        help="Path to the dictionary file.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate a random set of letters for the puzzle.",
    )
    parser.add_argument("--s0", type=str, default="", help="First set of letters.")
    parser.add_argument("--s1", type=str, default="", help="Second set of letters.")
    parser.add_argument("--s2", type=str, default="", help="Third set of letters.")
    parser.add_argument("--s3", type=str, default="", help="Fourth set of letters.")

    args = parser.parse_args()

    print("Building trie...")
    trie = build_trie_from_file(args.dictionary)
    print("Solving...")
    letter_sets = (args.s0, args.s1, args.s2, args.s3)

    if args.random or not all(letter_sets):
        letter_sets = generate_random_letter_set()
    print(f"Letter sets: {letter_sets}")
    solver = LetterBoxer(letter_sets, trie)
    ans = solver.solve()

    if not ans:
        print("No solutions found.")
        return

    print(f"Found {len(ans)} solutions of length {len(ans[0])}.")
    for sol in ans:
        print(sol)


if __name__ == "__main__":
    main()

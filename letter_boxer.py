import argparse
import pickle
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


def save_trie(trie, filename):
    """
    Save the trie to a file using pickle.
    """
    with open(filename, "wb") as f:
        pickle.dump(trie, f)


def load_trie(filename):
    """
    Load a trie from a file.
    """
    with open(filename, "rb") as f:
        return pickle.load(f)


@dataclass
class SolutionCandidate:
    char_set: ClassVar[FrozenSet[str]] = frozenset()
    word_list: List[str]
    residue: FrozenSet[str]

    def __init__(self, word_list: Optional[List[str]] = None) -> None:
        self.word_list = word_list if word_list is not None else []
        self.residue = self.calculate_residue()

    def calculate_residue(self) -> FrozenSet[str]:
        remaining = set(self.char_set)
        for c in chain(*self.word_list):
            if c in remaining:
                remaining.remove(c)
        return frozenset(remaining)


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

    def solve(self):
        """
        Solve the puzzle with branch and bound.
        """
        solutions = []
        best = None
        pq = []
        candidates_considered = 0

        for word in self.possible_words:
            heappush(pq, PrioritizedSolutionCandidate(SolutionCandidate([word])))

        progress_bar = tqdm(total=1, desc="Processing nodes", dynamic_ncols=True)

        while pq:
            curr = heappop(pq)
            candidates_considered += 1

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
                if (
                    next_word != curr.candidate.word_list[-1]
                ):  # no point in repeating the same word
                    new_candidate = SolutionCandidate(
                        curr.candidate.word_list + [next_word]
                    )
                    heappush(pq, PrioritizedSolutionCandidate(new_candidate))

        return solutions


def main():
    parser = argparse.ArgumentParser(
        description="Find optimal solutions to the Letter Boxer puzzle.",
        usage="%(prog)s --dictionary <path> ryd ivf wpn oge",
    )
    parser.add_argument(
        "--dictionary",
        type=Path,
        default="/usr/share/dict/words",
        help="Path to the dictionary file.",
    )
    parser.add_argument("s0", type=str, help="First set of letters.")
    parser.add_argument("s1", type=str, help="Second set of letters.")
    parser.add_argument("s2", type=str, help="Third set of letters.")
    parser.add_argument("s3", type=str, help="Fourth set of letters.")

    args = parser.parse_args()

    print("Building trie...")
    trie = build_trie_from_file(args.dictionary)
    print("Solving...")
    letter_sets = (args.s0, args.s1, args.s2, args.s3)
    solver = LetterBoxer(letter_sets, trie)
    ans = solver.solve()
    print(f"Found {len(ans)} solutions with {len(solver.possible_words)} words.")
    for sol in ans:
        print(sol)


if __name__ == "__main__":
    main()

from collections import defaultdict
import string
from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import chain, pairwise
from pathlib import Path
from typing import ClassVar, Dict, FrozenSet, List, Optional, Set, Tuple

from datrie import Trie
from tqdm import tqdm
import click


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
    residue: FrozenSet[str]

    def __init__(
        self,
        word_list: Optional[List[str]] = None,
        residue: Optional[FrozenSet[str]] = None,
    ) -> None:
        self.word_list = word_list if word_list is not None else []
        self.residue = (
            residue
            if residue is not None
            else frozenset(self.char_set - set("".join(self.word_list)))
        )

    def add_word(self, word: str) -> "SolutionCandidate":
        return SolutionCandidate(
            self.word_list + [word],
            self.residue - set(word),
        )

    def taxonomy(self) -> Tuple[FrozenSet[str], str]:
        return self.residue, self.word_list[-1][-1]

    def score(self) -> int:
        return len(self.word_list)


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
    g_score: Dict[str, Dict[FrozenSet[str], int]]
    h_score: Dict[str, Dict[FrozenSet[str], int]]

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

    def _get_congruent_scores(self, last_char, res):
        for other_res in self.g_score[last_char]:
            if res.issuperset(other_res):
                yield self.g_score[last_char][other_res]

    def _is_redundant_state(self, last_char, res, score):
        if self.find_all:
            return any(
                other_score < score
                for other_score in self._get_congruent_scores(last_char, res)
            )
        else:
            return any(
                other_score <= score
                for other_score in self._get_congruent_scores(last_char, res)
            )

    def _trim_redundant_states(self, last_char, res, score):
        to_remove = set()
        for other_res, other_score in self.g_score[last_char].items():
            if res.issubset(other_res) and score < other_score:
                to_remove.add(other_res)
        for other_res in to_remove:
            del self.g_score[last_char][other_res]

    def _update_h_score(self, solution: SolutionCandidate):
        res = self.charset
        for i, word in enumerate(solution.word_list):
            res = res - set(word)
            self.h_score[word[-1]][res] = min(
                self.h_score[word[-1]][res], len(solution.word_list) - i - 1
            )

    def solve(self, find_all: bool = False):
        """
        Solve the puzzle with branch and bound.
        """
        solutions = []
        best = None
        pq = []
        candidates_considered = 0
        self.find_all = find_all

        self.g_score = defaultdict(lambda: defaultdict(int))

        for word in self.possible_words:
            heappush(pq, PrioritizedSolutionCandidate(SolutionCandidate([word])))

        progress_bar = tqdm(total=1, desc="Processing nodes", dynamic_ncols=True)

        while pq:
            curr = heappop(pq)
            candidates_considered += 1

            res, last_char = curr.candidate.taxonomy()
            score = curr.candidate.score()

            if self._is_redundant_state(last_char, res, score):
                # we've visited a state with the same ending letter and a subset of the remaining letters with a better score
                #
                continue

            self.g_score[last_char][res] = score
            self._trim_redundant_states(last_char, res, score)

            # Update the progress bar based on the ratio of candidates considered to total candidates
            progress = candidates_considered / (candidates_considered + len(pq))
            progress_bar.update(progress - progress_bar.n)

            if curr.candidate.residue == frozenset():
                if best is None or len(curr.candidate.word_list) < len(best.word_list):
                    best = curr.candidate
                    if find_all:
                        pq = list(
                            filter(
                                lambda x: len(x.candidate.word_list)
                                <= len(best.word_list),
                                pq,
                            )
                        )
                    if not find_all:
                        pq = list(
                            filter(
                                lambda x: len(x.candidate.word_list)
                                < len(best.word_list),
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

            if (
                not find_all
                and best is not None
                and len(curr.candidate.word_list) >= len(best.word_list) - 1
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


@click.command()
@click.option(
    "--random", is_flag=True, help="Generate a random set of letters for the puzzle."
)
@click.option(
    "--dictionary",
    type=click.Path(exists=True),
    default="/usr/share/dict/words",
    help="Path to the dictionary file.",
)
@click.option(
    "--find-all",
    "find_all",
    is_flag=True,
    help="Find all optimal solutions, rather than just one.",
    default=False,
)
@click.argument("letters", nargs=-1)
def main(random, dictionary, find_all, letters):
    """
    Find optimal solutions to the Letter Boxer puzzle.
    """
    if random:
        letters = generate_random_letter_set()
    elif not letters:
        click.echo(
            "Error: You must provide a set of letters or use the --random option.",
            err=True,
        )
        return

    print("Building trie...")
    trie = build_trie_from_file(dictionary)
    print("Solving...")
    solver = LetterBoxer(letters, trie)
    ans = solver.solve(find_all=find_all)
    if not ans:
        print("No solutions found.")
        return

    if find_all:
        print(f"Found {len(ans)} solutions of length {len(ans[0])}.")
    elif len(ans) == 1:
        print("Found a solution:")

    for sol in ans:
        print(sol)

    if len(ans) > 1:
        print("Shortest solution:")
        print(sorted(ans, key=lambda a: sum(map(len, a)))[0])
        print("Longest solution:")
        print(sorted(ans, key=lambda a: -sum(map(len, a)))[0])


if __name__ == "__main__":
    main()

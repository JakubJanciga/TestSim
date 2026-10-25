"""
Monte Carlo simulation: how much of a test can a triplet of students pass?

Setup
-----
- The test has N questions and contains all of them.
- There are K students in the class (K < N/3).
- Each student learns L questions (L < N), chosen at random from the N.
  No two students learned exactly the same set of L questions.
- Students write the test in triplets. A question is passed if at least
  one of the three students in the triplet learned it.

The whole class experiment is repeated many times (default 10 000) and the
script reports the mean percentage of passed questions per triplet.

Usage
-----
    python3 triplet_test_sim.py
    python3 triplet_test_sim.py --questions 60 --students 15 --learned 20 --runs 10000
"""

import argparse
import math
import random
import statistics


def check_parameters(n_questions, n_students, n_learned):
    """Stop with a clear message if the parameters break the rules."""
    if not 0 < n_learned < n_questions:
        raise SystemExit("Error: L must satisfy 0 < L < N.")
    if n_students < 3:
        raise SystemExit("Error: K must be at least 3 to form a triplet.")
    if not n_students < n_questions / 3:
        raise SystemExit("Error: K must be smaller than N/3.")
    if n_students > math.comb(n_questions, n_learned):
        raise SystemExit("Error: there are not enough different question sets for K students.")


def assign_questions(n_questions, n_students, n_learned):
    """Give every student a random set of L questions. All sets are different."""
    all_questions = range(n_questions)
    students = []
    used_sets = set()

    while len(students) < n_students:
        learned = frozenset(random.sample(all_questions, n_learned))
        if learned in used_sets:
            continue  # same set as another student: draw again
        used_sets.add(learned)
        students.append(learned)

    return students


def form_triplets(students):
    """Shuffle the class and split it into groups of three.

    If K is not divisible by 3, the leftover one or two students sit out.
    """
    shuffled = list(students)
    random.shuffle(shuffled)
    n_triplets = len(shuffled) // 3
    return [shuffled[3 * i : 3 * i + 3] for i in range(n_triplets)]


def percent_passed(triplet, n_questions):
    """Percentage of the test the triplet can answer together."""
    a, b, c = triplet
    known_by_someone = a | b | c
    return 100 * len(known_by_someone) / n_questions


def run_one_class(n_questions, n_students, n_learned):
    """Simulate one class writing the test. Returns one percentage per triplet."""
    students = assign_questions(n_questions, n_students, n_learned)
    triplets = form_triplets(students)
    return [percent_passed(t, n_questions) for t in triplets]


def exact_expected_percent(n_questions, n_learned):
    """Exact expected percentage, used as a check on the simulation.

    A triplet holds 3 different sets, drawn uniformly from all C(N, L) sets.
    A given question is missed only if none of the 3 sets contains it.
    There are C(N-1, L) sets without that question, so:
        P(missed) = C(M0, 3) / C(M, 3),  M = C(N, L),  M0 = C(N-1, L)
    """
    m = math.comb(n_questions, n_learned)
    m0 = math.comb(n_questions - 1, n_learned)
    p_missed = (m0 / m) * ((m0 - 1) / (m - 1)) * ((m0 - 2) / (m - 2))
    return 100 * (1 - p_missed)


def main():
    parser = argparse.ArgumentParser(description="Monte Carlo simulation of triplets writing a test.")
    parser.add_argument("--questions", "-N", type=int, default=60, help="N: number of questions on the test")
    parser.add_argument("--students", "-K", type=int, default=15, help="K: number of students in the class")
    parser.add_argument("--learned", "-L", type=int, default=20, help="L: questions each student learns")
    parser.add_argument("--runs", type=int, default=10_000, help="number of simulated classes")
    parser.add_argument("--seed", type=int, default=None, help="random seed (for repeatable results)")
    args = parser.parse_args()

    n, k, l = args.questions, args.students, args.learned
    check_parameters(n, k, l)
    random.seed(args.seed)

    all_triplet_results = []
    for _ in range(args.runs):
        all_triplet_results.extend(run_one_class(n, k, l))

    mean = statistics.mean(all_triplet_results)
    spread = statistics.stdev(all_triplet_results)
    std_error = spread / math.sqrt(len(all_triplet_results))

    print(f"N = {n} questions, K = {k} students, L = {l} learned each")
    print(f"{args.runs} simulated classes, {k // 3} triplets per class "
          f"({len(all_triplet_results)} triplets in total)")
    if k % 3:
        print(f"Note: {k % 3} student(s) per class sit out (K is not divisible by 3)")
    print()
    print(f"Mean % of passed questions per triplet: {mean:.2f} %  (± {1.96 * std_error:.2f} %, 95% CI)")
    print(f"Standard deviation between triplets:   {spread:.2f} %")
    print(f"Lowest / highest triplet:              {min(all_triplet_results):.1f} % / {max(all_triplet_results):.1f} %")
    print(f"Exact expected value (formula check):  {exact_expected_percent(n, l):.2f} %")


if __name__ == "__main__":
    main()

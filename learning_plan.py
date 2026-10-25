"""
Find a coordinated learning plan for a class.

Setup (same as triplet_test_sim.py)
-----------------------------------
- The test has N questions and contains all of them.
- There are K students; each student learns L questions.
- No two students learn exactly the same set of questions.
- Students are put into triplets at random. A question is passed if at
  least one of the three students learned it.

Goal
----
Instead of every student picking questions at random, the class agrees on
who learns what, so that EVERY possible triplet passes at least TARGET %
of the test, no matter how the triplets are drawn.

How the search works
--------------------
1. Start from a balanced plan: the questions are dealt out round-robin,
   so every question is learned by (almost) the same number of students.
2. Repeatedly pick two students and let them swap one question each.
   Keep the swap unless it makes the plan worse. A swap never changes
   how many students learn each question, so the plan stays balanced.
3. Stop when every triplet reaches the target and all students have
   different sets, or when the step limit is reached.

If L is not given, the script skips every L that provably cannot work
(see could_work) and tries L, L+1, ... until it finds a plan.

Note: "impossible" is proven by counting; "no plan found" only means the
search gave up, not that no such plan exists.

Usage
-----
    python3 learning_plan.py -N 15 -K 30             # find the smallest L
    python3 learning_plan.py -N 15 -K 30 -L 11       # plan for a given L
    python3 learning_plan.py -N 100 -K 30 -L 58 --target 90

Large cases are slower; tight ones (e.g. N=100, L=56) may need --steps 100000.
"""

import argparse
import csv
import itertools
import math
import random
from collections import Counter


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def needed_questions(n_questions, target_percent):
    """How many questions a triplet must pass to reach the target."""
    return math.ceil(n_questions * target_percent / 100 - 1e-9)


def balanced_mean_percent(n_questions, n_students, n_learned):
    """Best possible MEAN score per triplet (every question learned by
    the same number of students). No plan can do better on average."""
    total = n_students * n_learned
    low = total // n_questions             # most questions: learned by `low` students
    n_high = total - low * n_questions     # this many: learned by low + 1 students
    all_triplets = math.comb(n_students, 3)
    missed = (n_high * math.comb(n_students - low - 1, 3)
              + (n_questions - n_high) * math.comb(n_students - low, 3)) / all_triplets
    return 100 * (n_questions - missed) / n_questions


def could_work(n_questions, n_students, n_learned, needed):
    """Two quick tests that rule out values of L that can never work.

    1. Even the best plan cannot beat the balanced mean, and the mean must
       reach the target if every single triplet does.
    2. A triplet may miss at most m = N - needed questions. So for any m + 1
       questions, at most 2 students may skip all of them (3 such students
       would form a triplet below the target). Counting over all groups of
       m + 1 questions gives:  K * C(N - L, m + 1) <= 2 * C(N, m + 1).
    """
    best_mean = balanced_mean_percent(n_questions, n_students, n_learned)
    if best_mean < 100 * needed / n_questions:
        return False
    m = n_questions - needed
    skipped_groups = n_students * math.comb(n_questions - n_learned, m + 1)
    return skipped_groups <= 2 * math.comb(n_questions, m + 1)


def smallest_possible_l(n_questions, n_students, needed):
    """Smallest L that passes both quick tests."""
    for n_learned in range(1, n_questions):
        if could_work(n_questions, n_students, n_learned, needed):
            return n_learned
    return n_questions - 1


# ---------------------------------------------------------------------------
# The plan and how good it is
# ---------------------------------------------------------------------------

def balanced_start(n_questions, n_students, n_learned):
    """Deal questions round-robin: student 1 gets the first L questions,
    student 2 the next L, and so on, wrapping around after question N."""
    plan = []
    for student in range(n_students):
        first = student * n_learned
        plan.append({(first + j) % n_questions for j in range(n_learned)})
    return plan


def passed_by(triplet, plan):
    """Number of questions the triplet can answer together."""
    a, b, c = triplet
    return len(plan[a] | plan[b] | plan[c])


def shortfall(triplets, plan, needed):
    """Total number of questions the triplets fall short of the target."""
    return sum(max(0, needed - passed_by(t, plan)) for t in triplets)


def duplicates(plan):
    """How many students learn exactly the same set as someone else."""
    distinct_sets = {frozenset(s) for s in plan}
    return len(plan) - len(distinct_sets)


def swap(plan, i, j, a, b):
    """Student i hands question a to student j and takes question b instead."""
    plan[i].remove(a)
    plan[i].add(b)
    plan[j].remove(b)
    plan[j].add(a)


def try_random_swap(plan, triplets_of, needed):
    """Try one random swap. Undo it if it makes things worse.
    Returns the change in score (0 or negative)."""
    i, j = random.sample(range(len(plan)), 2)
    only_i = list(plan[i] - plan[j])
    only_j = list(plan[j] - plan[i])
    if not only_i or not only_j:
        return 0
    a = random.choice(only_i)
    b = random.choice(only_j)

    # Only triplets containing student i or j can change.
    affected = triplets_of[i] | triplets_of[j]
    before = shortfall(affected, plan, needed) + duplicates(plan)
    swap(plan, i, j, a, b)
    after = shortfall(affected, plan, needed) + duplicates(plan)

    if after > before:
        swap(plan, i, j, b, a)  # undo
        return 0
    return after - before


def find_plan(n_questions, n_students, n_learned, needed, max_steps):
    """Search for a plan where every triplet passes at least `needed`
    questions. Returns the plan, or None if the search gives up."""
    plan = balanced_start(n_questions, n_students, n_learned)
    triplets = list(itertools.combinations(range(n_students), 3))
    triplets_of = {s: {t for t in triplets if s in t} for s in range(n_students)}

    score = shortfall(triplets, plan, needed) + duplicates(plan)
    for _ in range(max_steps):
        if score == 0:
            return plan
        score += try_random_swap(plan, triplets_of, needed)
    return plan if score == 0 else None


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_plan(plan, filename):
    """Write one row per student, questions numbered from 1."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["student", "questions_to_learn"])
        for student, questions in enumerate(plan, start=1):
            numbers = " ".join(str(q + 1) for q in sorted(questions))
            writer.writerow([student, numbers])


def print_report(plan, n_questions):
    """Show how every possible triplet would score with this plan."""
    triplets = list(itertools.combinations(range(len(plan)), 3))
    scores = [passed_by(t, plan) for t in triplets]
    learners = Counter(q for s in plan for q in s)

    fewest, most = min(learners.values()), max(learners.values())
    if fewest == most:
        print(f"Every question is learned by {fewest} students")
    else:
        print(f"Every question is learned by {fewest}-{most} students")
    print(f"All {len(triplets)} possible triplets checked:")
    for passed, count in sorted(Counter(scores).items()):
        print(f"  {passed:>3}/{n_questions} questions "
              f"({100 * passed / n_questions:5.1f} %): {count} triplets")
    print(f"Mean per triplet: {100 * sum(scores) / len(scores) / n_questions:.2f} %")


def main():
    parser = argparse.ArgumentParser(description="Find a learning plan so every triplet reaches a target score.")
    parser.add_argument("--questions", "-N", type=int, default=15, help="N: number of questions on the test")
    parser.add_argument("--students", "-K", type=int, default=30, help="K: number of students in the class")
    parser.add_argument("--learned", "-L", type=int, default=None, help="L: questions each student learns (omit to find the smallest)")
    parser.add_argument("--target", type=float, default=90, help="minimum %% every triplet must pass")
    parser.add_argument("--steps", type=int, default=20_000, help="search steps per value of L")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    args = parser.parse_args()

    n, k = args.questions, args.students
    if k < 3:
        raise SystemExit("Error: need at least 3 students.")
    random.seed(args.seed)
    needed = needed_questions(n, args.target)
    print(f"N = {n} questions, K = {k} students, target {args.target:g} % "
          f"= at least {needed}/{n} questions for every triplet\n")

    if args.learned is not None:
        values_to_try = [args.learned]
    else:
        first = smallest_possible_l(n, k, needed)
        if first > 1:
            print(f"L = 1 to {first - 1}: impossible (ruled out by counting, no search needed)")
        values_to_try = range(first, n)

    for n_learned in values_to_try:
        if not could_work(n, k, n_learned, needed):
            print(f"L = {n_learned}: impossible (ruled out by counting)")
            continue
        if math.comb(n, n_learned) < k:
            print(f"L = {n_learned}: not enough different question sets for {k} students")
            continue
        plan = find_plan(n, k, n_learned, needed, args.steps)
        if plan is None:
            print(f"L = {n_learned}: no plan found in {args.steps} steps")
            continue

        print(f"L = {n_learned}: plan found\n")
        print_report(plan, n)
        filename = f"plan_N{n}_K{k}_L{n_learned}.csv"
        save_plan(plan, filename)
        print(f"\nPlan saved to {filename}")
        return

    print("\nNo plan found. If the search gave up, try more --steps or another --seed;"
          " otherwise try a larger L.")


if __name__ == "__main__":
    main()

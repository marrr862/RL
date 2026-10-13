"""
=====================================================================
 ASSIGNMENT: Multi-Armed Bandits for Headline Recommendation
=====================================================================
Scenario:
  A news platform has K = 6 candidate headlines for its home screen.
  Every visiting user is shown ONE headline (= pulling one arm).
  Reward: 1 if the user clicks, 0 otherwise.
  Goal: maximise total clicks (i.e. minimise cumulative regret).

What is given:     the environment, the A/B-test baseline, Epsilon-Greedy,
                   the simulation loop and the plotting code.
What YOU write:    UCB1 and Thompson Sampling (Part A), plus one
                   extension of your choice (Part C).
Search for "TODO" to find every place you need to write code.

Requirements:  pip install numpy matplotlib
Run:           python mab_assignment.py
Self-check:    python check_my_work.py
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


STUDENT_ID = 36127


def make_true_ctr(student_id, k=6):
    """Generate k hidden CTRs from the student ID. Exactly one headline is
    clearly best (1.5 to 3 percentage points above the runner-up)."""
    rng = np.random.default_rng(student_id)
    others = np.round(rng.uniform(0.02, 0.07, size=k - 1), 3)
    best = round(float(others.max() + rng.uniform(0.015, 0.03)), 3)
    ctr = np.append(others, best)
    rng.shuffle(ctr)
    return [float(c) for c in ctr]


class HeadlineEnvironment:
    """Each headline has a hidden true click-through rate (CTR).
    Showing headline k returns 1 (click) with probability true_ctr[k]."""

    def __init__(self, true_ctr, seed=None):
        self.true_ctr = np.array(true_ctr, dtype=float)
        self.k = len(true_ctr)
        self.rng = np.random.default_rng(seed)

    @property
    def best_ctr(self):
        return self.true_ctr.max()

    def step(self, t):
        """Called once per user before the headline is chosen.
        Stationary world: nothing changes."""
        pass

    def show_headline(self, arm):
        """Simulate one user seeing headline `arm`. Returns 0 or 1."""
        return int(self.rng.random() < self.true_ctr[arm])


class BreakingNewsEnvironment(HeadlineEnvironment):
    """Non-stationary world for Extension C4: at user `swap_at`, the best
    and the worst headline swap their CTRs (breaking news happened)."""

    def __init__(self, true_ctr, swap_at, seed=None):
        super().__init__(true_ctr, seed)
        self.swap_at = swap_at

    def step(self, t):
        if t == self.swap_at:
            b, w = np.argmax(self.true_ctr), np.argmin(self.true_ctr)
            self.true_ctr[b], self.true_ctr[w] = self.true_ctr[w], self.true_ctr[b]


class BaseAgent:
    """GIVEN. Keeps a count and a running-average CTR for every arm."""

    def __init__(self, k):
        self.k = k
        self.counts = np.zeros(k)
        self.values = np.zeros(k)

    def update(self, arm, reward):

        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n


class UniformABTest(BaseAgent):
    """GIVEN. Baseline: cycle through all headlines equally (A/B/n test)."""
    name = "A/B Test (uniform)"

    def __init__(self, k):
        super().__init__(k)
        self.t = 0

    def select_arm(self):
        arm = self.t % self.k
        self.t += 1
        return arm


class EpsilonGreedy(BaseAgent):
    """GIVEN. With prob. epsilon explore a random arm, otherwise exploit."""

    def __init__(self, k, epsilon=0.1, seed=None):
        super().__init__(k)
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)
        self.name = f"Epsilon-Greedy (eps={epsilon})"

    def select_arm(self):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.k))
        best = np.flatnonzero(self.values == self.values.max())
        return int(self.rng.choice(best))


class UCB1(BaseAgent):
    """Upper Confidence Bound.

    Rule: pick the arm that maximises   mu_i + sqrt(c * ln(t) / N_i)
      mu_i  = self.values[i]  (observed mean CTR of arm i)
      N_i   = self.counts[i]  (times arm i has been shown)
      t     = total number of users served so far (count this yourself)
      c     = exploration constant (c = 2 is standard UCB1)

    Before using the formula, every arm must be shown once
    (otherwise N_i = 0 and you divide by zero).
    select_arm() must return a plain Python int.
    update() is inherited from BaseAgent - you do not need to write it.
    """

    def __init__(self, k, c=2.0):
        super().__init__(k)
        self.c = c
        self.name = "UCB1" if c == 2.0 else f"UCB1 (c={c})"
        self.t = 0

    def select_arm(self):
        self.t += 1
        untried = np.flatnonzero(self.counts == 0)
        if untried.size:
            return int(untried[0])
        bonus = np.sqrt(self.c * np.log(self.t) / self.counts)
        return int(np.argmax(self.values + bonus))


class ThompsonSampling(BaseAgent):
    """Bayesian Thompson Sampling with Beta-Bernoulli beliefs.

    Keep two arrays of length k:
      self.alpha = 1 + number of clicks      (start at all ones)
      self.beta  = 1 + number of non-clicks  (start at all ones)
    select_arm(): draw ONE sample from Beta(alpha_i, beta_i) for every arm
                  (use self.rng.beta) and return the index of the largest.
    update():     call super().update(arm, reward) so counts/values stay
                  correct, then update alpha and beta for that arm.
    select_arm() must return a plain Python int.
    """
    name = "Thompson Sampling"

    def __init__(self, k, seed=None):
        super().__init__(k)
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(k)
        self.beta = np.ones(k)

    def select_arm(self):
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm, reward):
        super().update(arm, reward)
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward


def run_experiment(agent, env, n_users):
    """Serve n_users visitors. Returns per-step regret and chosen arms."""
    regrets = np.zeros(n_users)
    choices = np.zeros(n_users, dtype=int)
    for t in range(n_users):
        env.step(t)
        arm = agent.select_arm()
        reward = env.show_headline(arm)
        agent.update(arm, reward)
        regrets[t] = env.best_ctr - env.true_ctr[arm]
        choices[t] = arm
    return regrets, choices


def compare(agent_factories, true_ctr, n_users=20_000, n_runs=20,
            env_factory=None, title="", filename="results.png"):
    """Run every agent n_runs times, print a summary table, save plots.

    agent_factories: list of functions  seed -> new agent
    env_factory:     function (true_ctr, seed) -> environment
                     (default: a normal HeadlineEnvironment)
    Returns a dict  name -> final average cumulative regret.
    """
    if env_factory is None:
        env_factory = lambda ctr, seed: HeadlineEnvironment(ctr, seed=seed)
    k = len(true_ctr)
    names = [f(0).name for f in agent_factories]
    cum_regret = {n: np.zeros(n_users) for n in names}
    share = {n: np.zeros(k) for n in names}
    optimal = {n: 0.0 for n in names}

    for run in range(n_runs):
        for make in agent_factories:
            agent = make(run)
            env = env_factory(true_ctr, 1000 + run)
            regrets, choices = run_experiment(agent, env, n_users)
            cum_regret[agent.name] += np.cumsum(regrets) / n_runs
            share[agent.name] += np.bincount(choices, minlength=k) / (n_users * n_runs)
            optimal[agent.name] += np.mean(regrets == 0) / n_runs

    print(f"\n{title}")
    print(f"{'Strategy':<32}{'Regret':>10}{'% on best':>12}")
    print("-" * 54)
    for n in names:
        print(f"{n:<32}{cum_regret[n][-1]:>10.1f}{100 * optimal[n]:>11.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for n in names:
        axes[0].plot(cum_regret[n], label=n)
    axes[0].set_title("Cumulative regret (lower is better)")
    axes[0].set_xlabel("Users served")
    axes[0].set_ylabel("Expected clicks lost")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    width = 0.8 / len(names)
    x = np.arange(k)
    for i, n in enumerate(names):
        axes[1].bar(x + i * width, 100 * share[n], width, label=n)
    axes[1].set_xticks(x + 0.4 - width / 2)
    axes[1].set_xticklabels([f"H{i}\n{c}" for i, c in enumerate(true_ctr)])
    axes[1].set_title("Share of traffic per headline (initial CTRs)")
    axes[1].set_ylabel("% of users")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Saved plot -> {filename}")
    return {n: cum_regret[n][-1] for n in names}


if __name__ == "__main__":
    TRUE_CTR = make_true_ctr(STUDENT_ID)
    K = len(TRUE_CTR)
    print(f"Student ID {STUDENT_ID} -> TRUE_CTR = {TRUE_CTR}")
    print(f"Best headline: H{int(np.argmax(TRUE_CTR))}")


    compare(
        [lambda s: UniformABTest(K),
         lambda s: EpsilonGreedy(K, epsilon=0.1, seed=s),
         lambda s: UCB1(K),
         lambda s: ThompsonSampling(K, seed=s)],
        TRUE_CTR, title="Part B: all strategies", filename="part_b.png")


    compare(
        [lambda s: UCB1(K, c=0.5),
         lambda s: UCB1(K, c=2.0),
         lambda s: UCB1(K, c=5.0)],
        TRUE_CTR, title="C2: UCB1 constant", filename="part_c2.png")

"""One-factor ablation, three training seeds; does not modify given functions."""
import json
import numpy as np
import sarsa_ql_assignment as m

desc, slip = m.make_student_config(m.STUDENT_ID)
env = m.MyFrozenLake(desc, slip)
Q_star, _ = m.value_iteration(env.P, 16)
v_star = Q_star.max(axis=1)[env.start_state]
lab = dict(num_episodes=500, alpha=.1, gamma=.99, initial_epsilon=1., min_epsilon=.01, decay_rate=.995)
settings = {'lab':lab,'more episodes only':dict(lab,num_episodes=30000),
            'slower decay only':dict(lab,decay_rate=.9997),
            'smaller alpha only':dict(lab,alpha=.02),'all tuned':m.HYPERPARAMS}
results = {}
for label, params in settings.items():
    results[label] = {}
    for name, algo in [('SARSA',m.sarsa_learning),('Q-learning',m.q_learning)]:
        ratios = []
        for seed in range(3):
            Q,_=algo(env,seed=seed,**params)
            ratios.append(float(m.policy_value(env.P,Q.argmax(axis=1))[env.start_state]/v_star))
        results[label][name]={'ratios':ratios,'mean':float(np.mean(ratios))}
print(json.dumps(results,indent=2))

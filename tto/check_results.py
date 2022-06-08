import glob
import json
import numpy as np

def report(env_name):
    scores = []
    for fname in glob.glob("logs/{env_name}/plan/gpt/*/0/rollout.json".format(env_name=env_name)):
        with open(fname, "r") as f:
            data = json.loads(f.read())
            scores.append(data["score"])
    print(env_name)
    print(scores)
    print("average +- stderr: {:.3f} +- {:.3f}".format(np.mean(scores), np.std(scores) / (15 ** 0.5)))

# report("halfcheetah-medium-v2")
report("hopper-medium-v2")
# report("walker2d-medium-v2")


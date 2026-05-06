import numpy as np
from trainer import train

def leave_one_subject_out(actions,subjects):

    unique=np.unique(subjects)

    results=[]

    for s in unique:

        train_idx=subjects!=s
        test_idx=subjects==s

        train_actions=actions[train_idx]
        test_actions=actions[test_idx]

        q=train(train_actions)

        results.append(q)

    return results
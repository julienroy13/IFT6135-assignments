import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pdb



with open("inputs.pkl", 'rb') as f:
    inputs = pickle.load(f)


plt.figure(figsize=(40, 20))
plt.imshow(inputs)
plt.title('Inputs', fontweight='bold', fontsize=30)

plt.savefig('inputs.png', bbox_inches='tight')
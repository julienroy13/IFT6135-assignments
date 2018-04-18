import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pdb



with open("attentions_read.pkl", 'rb') as f:
    reads = pickle.load(f)

with open("attentions_write.pkl", 'rb') as f:
    writes = pickle.load(f)

#pdb.set_trace()


plt.figure(figsize=(40, 20))

plt.subplot(1,2,2)
plt.imshow(reads)
plt.title('Read attention weights', fontweight='bold', fontsize=30)

plt.subplot(1,2,1)
plt.imshow(writes)
plt.title('Write attention weights', fontweight='bold', fontsize=30)

plt.savefig('attention_weights.png', bbox_inches='tight')
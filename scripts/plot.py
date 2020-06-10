import numpy as np
import matplotlib.pyplot as plt

test = np.load('/gpfs/alpine/world-shared/stf011/atsaris/gnn_results_md_out/mpnn_bzrmd/summaries_0.npz')
print(test.files)

# Loss
plt.plot(test['train_loss'])
plt.plot(test['valid_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Acc
plt.plot(test['valid_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['test'], loc='upper left')
plt.show()

# Time
plt.plot(test['train_time'])
plt.plot(test['valid_time'])
plt.title('Time')
plt.ylabel('Time')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#print(test['epoch'])
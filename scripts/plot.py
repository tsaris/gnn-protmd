import numpy as np
import matplotlib.pyplot as plt

test = np.load('/gpfs/alpine/world-shared/stf011/atsaris/gnn_results_md_out/G12D_r0_ch2_temporal/summaries_0.npz')
print(test.files)

# Loss
plt.plot(test['train_loss'])
plt.plot(test['valid_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Acc
plt.plot(test['valid_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val'], loc='upper left')
plt.show()

exit(-1)
# Time
plt.plot(test['train_time'])
plt.plot(test['valid_time'])
plt.title('Time (sec)')
plt.ylabel('Time (sec)')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#print(test['epoch'])

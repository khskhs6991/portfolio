import numpy as np
from matplotlib import pyplot as plt

f1_ori = [80.12, 85.04, 87.68]
f1_add = [81.44, 82.13, 91.65]
f1_concat = [81.13, 84.73, 86.89]
f1_asc = [85.5, 87.72, 92.46]

auroc_ori = [66.48, 68.64, 73.29]
auroc_add = [68.03, 67.7, 75.08]
auroc_concat = [66.7, 69.61, 74.2]
auroc_asc = [61.01, 68.24, 76.36]

time_ori = [753, 2246, 3038]
time_add = [758, 2138, 3064]
time_concat = [761, 2764, 3136]
time_asc = [765, 2834, 3246]

# dagmm

# f1_ori = [74.11, 76.59, 81.24]
# f1_add = [74.29, 81.63, 83.48]
# f1_concat = [75.41, 77.69, 85.34]
# f1_asc = [77.32, 84.69, 87.33]

# auroc_ori = [60.1, 65.85, 74.55]
# auroc_add = [59.08, 62.15, 73.17]
# auroc_concat = [56.49, 65.7, 76.93]
# auroc_asc = [58, 61.96, 77.61]

# time_ori = [86.96, 333.78, 584.83]
# time_add = [86.26, 337.52, 581.8]
# time_concat = [89.66, 348.56, 609.13]
# time_asc = [100.14, 399.31, 686.38]

# dsvdd

x = [0.1, 0.4, 0.7]

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(x, time_ori, marker='o', linestyle='-', color='b')
plt.xlabel('Train ratio',fontsize=10)
plt.ylabel('Time (s)',fontsize=10)
plt.grid()
plt.subplot(2, 2, 2)
plt.plot(x, time_add, marker='o', linestyle='-', color='r')
plt.xlabel('Train ratio',fontsize=10)
plt.ylabel('Time (s)',fontsize=10)
plt.grid()
plt.subplot(2, 2, 3)
plt.plot(x, time_concat, marker='o', linestyle='-', color='c')
plt.xlabel('Train ratio',fontsize=10)
plt.ylabel('Time (s)',fontsize=10)
plt.grid()
plt.subplot(2, 2, 4)
plt.plot(x, time_asc, marker='o', linestyle='-', color='y')
plt.xlabel('Train ratio',fontsize=10)
plt.ylabel('Time (s)',fontsize=10)
plt.grid()
plt.show()

# f1_dagmm = [87.68, 91.65, 86.89, 92.46]
# auroc_dagmm = [73.29, 75.08, 74.2, 76.36]
# time_dagmm = [3038, 3064, 3136, 3246]

# f1_dsvdd = [81.24, 83.48, 85.34, 87.33]
# auroc_dsvdd = [74.55, 73.17, 76.93, 77.61]
# time_dsvdd = [581.83, 584.8, 609.13, 686.38]

# label = ['Original', 'Add', 'Concat', 'Proposed']
# x = np.arange(4)

# plt.figure(figsize=(15, 5))
# plt.subplot(1, 3, 1)
# plt.bar(x, f1_dagmm, color='green', alpha=0.5)
# plt.plot(x, f1_dagmm, marker='o', color='b', linestyle='--')
# plt.xticks(x, label)
# plt.xlabel('skip connection type')
# plt.ylabel('F1 score (%)')

# plt.subplot(1, 3, 2)
# plt.bar(x, auroc_dagmm, color='r', alpha=0.5)
# plt.plot(x, auroc_dagmm, marker='o', color='b', linestyle='--')
# plt.xticks(x, label)
# plt.xlabel('skip connection type')
# plt.ylabel('AUROC')
# plt.subplot(1, 3, 3)
# plt.bar(x, time_dagmm, color='y', alpha=0.5)
# plt.plot(x, time_dagmm, marker='o', color='b', linestyle='--')
# plt.xticks(x, label)
# plt.xlabel('skip connection type')
# plt.ylabel('Time (s)')
# plt.show()
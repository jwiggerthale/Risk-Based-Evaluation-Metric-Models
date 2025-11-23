'''
this scrip implements comparison of wba for different models
Note: confusion matrices are defined manually based on the images saved when calling script train_model.py
'''

import numpy as np
import matplotlib.pyplot as plt


# function to calculate expected risl
def er(P: np.array, 
       C: np.array):
    assert P.shape == C.shape and P.shape[0] == P.shape[1]
    num_classes = P.shape[1]
    er = 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            er += P[i, j] * C[i, j]
    return er


# function to calculate wba
def wba(er: float, 
        er_worst: float):
    wba = 1 - er / er_worst
    return wba

# define confusion matrices and scale
cm_rn = np.array([
    [759, 158, 8], 
    [9, 571, 0],
    [416, 3, 645]
])

scaled_rn = cm_rn / cm_rn.sum()


cm_vgg = np.array([
    [785, 28, 112], 
    [147, 433, 0],
    [429, 9, 626]
])

scaled_vgg = cm_vgg / cm_vgg.sum()


cm_rxt = np.array([
    [850, 25, 50], 
    [130, 448, 2],
    [398, 1, 665]
])

scaled_rxt = cm_rxt / cm_rxt.sum()


acc_vgg  = scaled_vgg[0,0]  + scaled_vgg[1, 1] + scaled_vgg[2, 2]
acc_rxt = scaled_rxt[0,0]  + scaled_rxt[1, 1] + scaled_rxt[2, 2]
acc_rn = scaled_rn[0,0]  + scaled_rn[1, 1] + scaled_rn[2, 2]

print(f'RN: {acc_rn}; RXT: {acc_rxt}; VGG: {acc_vgg}')


# define cost matrices

c_1 = np.array([
    [0, 2, 5], 
    [7, 0, 3], 
    [10, 8, 0]
])

c_2 = np.array([
    [0, 10, 1], 
    [1, 0, 1], 
    [1, 1, 0]
])

# c1
## get maximum costs; get probability for each class and take maximum misclassification costs for this class as cost
probabilities = scaled_vgg.sum(axis = 1)
max_costs = probabilities[0] * c_1[0, 2] + probabilities[1] * c_1[1, 0] + probabilities[2] * c_1[2, 0]


er_real_vgg = er(P = scaled_vgg, 
                 C = c_1)
wba_vgg_real = wba(er = er_real_vgg, 
              er_worst = max_costs)

print(f'ER for VGG: {er_real_vgg}; WBA for VGG: {wba_vgg_real}')

er_real_rn = er(P = scaled_rn, 
                 C = c_1)
wba_rn_real = wba(er = er_real_rn, 
              er_worst = max_costs)

print(f'ER for RN: {er_real_rn}; WBA for VGG: {wba_rn_real}')


er_real_rxt = er(P = scaled_rxt, 
                 C = c_1)
wba_rxt_real = wba(er = er_real_rxt, 
              er_worst = max_costs)

print(f'ER for RN: {er_real_rxt}; WBA for VGG: {wba_rxt_real}')



# Calculation for c2
c_2 = np.array([
    [0, 10, 1], 
    [1, 0, 1], 
    [1, 1, 0]
])


# c1
## get maximum costs; get probability for each class and take maximum misclassification costs for this class as cost
probabilities = scaled_vgg.sum(axis = 1)
max_costs = probabilities[0] * c_2[0, 1] + probabilities[1] * c_2[1, 0] + probabilities[2] * c_2[2, 0]


er_real_vgg = er(P = scaled_vgg, 
                 C = c_2)
wba_vgg_real = wba(er = er_real_vgg, 
              er_worst = max_costs)

print(f'ER for VGG: {er_real_vgg}; WBA for VGG: {wba_vgg_real}')


er_real_rn = er(P = scaled_rn, 
                 C = c_2)
wba_rn_real = wba(er = er_real_rn, 
              er_worst = max_costs)

print(f'ER for RN: {er_real_rn}; WBA for RN: {wba_rn_real}')

er_real_rxt = er(P = scaled_rxt, 
                 C = c_2)
wba_rxt_real = wba(er = er_real_rxt, 
              er_worst = max_costs)

print(f'ER for RXT: {er_real_rxt}; WBA for RN: {wba_rxt_real}')




# get stats for c3
## c3 works with adaptive c for c_3[0, 1]
cs = np.arange(1, 11)

stats_vgg = {}
stats_rn = {}
stats_rxt = {}
for c in cs: 
    c_3 = np.array([
        [0, c, 1], 
        [1, 0, 1], 
        [1, 1, 0]
    ])
    probabilities = scaled_vgg.sum(axis = 1)
    max_costs = probabilities[0] * c_3[0, 1] + probabilities[1] * c_3[1, 0] + probabilities[2] * c_3[2, 0]
    er_rn = er(P = scaled_rn, 
                 C = c_3 )
    wba_rn = wba(er = er_rn, 
                er_worst = max_costs)
    er_vgg = er(P = scaled_vgg, 
                 C = c_3 )
    wba_vgg = wba(er = er_vgg, 
                er_worst = max_costs)
    er_rxt = er(P = scaled_rxt, 
                 C = c_3 )
    wba_rxt = wba(er = er_rxt, 
                er_worst = max_costs)
    stats_rn[c] = {'er': er_rn, 'wba': wba_rn}
    stats_rxt[c] = {'er': er_rxt, 'wba': wba_rxt}
    stats_vgg[c] = {'er': er_vgg, 'wba': wba_vgg}

plt.plot([k for k in stats_rn.keys()], [stats_rn[k]['wba'] for k in stats_rn.keys()], label = 'ResNet18')
plt.plot([k for k in stats_vgg.keys()], [stats_vgg[k]['wba'] for k in stats_vgg.keys()], label = 'VGG16')
plt.plot([k for k in stats_rxt.keys()], [stats_rxt[k]['wba'] for k in stats_rxt.keys()], label = 'ResNext50')
plt.legend()
plt.title(r'Developmet of WBA when varying $c_{N->P}$')
plt.xlabel(r'$c_{N->P}$')
plt.ylabel('WBA')
plt.savefig('WBA_CFN_VGG.png')

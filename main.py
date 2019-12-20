import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import func
import scipy.linalg


# G = nx.karate_club_graph()
#
# for u in G.nodes():
#     if G.nodes[u]['club'] == 'Mr. Hi':
#         G.nodes[u]['color'] = "#ababab"
#     else:
#         G.nodes[u]['color'] = "#000000"
#
# pos = nx.spring_layout(G)
# nx.draw_networkx(G, pos=pos, node_size=10, node_color=list(nx.get_node_attributes(G,'color').values()), with_labels=False, width=0.5)
# plt.show()

random.seed(110)
# G = nx.barbell_graph(10,5)
# G = nx.cycle_graph(10)
G = nx.star_graph(10)
# G = nx.wheel_graph(9)
# G = nx.gnp_random_graph(10,0.5)
# G = nx.watts_strogatz_graph(20, 8, 0.5)
# G = nx.barabasi_albert_graph(20, 5)

popsize = 10
n = G.number_of_nodes()

solution = []
fitcall = []
fit = dict()
fit['best'] = []
fit['median'] = []
fit['mean'] = []


numiter = 500

for ii in range(25):
    result = func.ga(popsize, n, 0.8, 0.1, numiter, G, 0.5)

    m = max(result['popfitness'])
    best = 0
    for k in range(popsize):
        if abs(result['popfitness'][best] - m) < (10 ** -6):
            best = k

    solution.append(result['population'][best])
    fitcall.append(result['fitnesscalls'])
    fit['best'].append(result['fitness']['best'][numiter-1])
    fit['mean'].append(result['fitness']['mean'][numiter-1])
    fit['median'].append(result['fitness']['median'][numiter-1])
    if ii < 3:
        plt.plot(range(0, numiter + 1), result['fitness']['best'], label="Best")
        plt.plot(range(0, numiter + 1), result['fitness']['mean'], label="Mean")
        plt.plot(range(0, numiter + 1), result['fitness']['median'], label="Median")
        plt.title("Star Graph")
        plt.xlabel("Generation Number")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()

bxdata = [fit['best'], fit['median'], fit['mean']]
fig = plt.figure(1, figsize=(9, 6))
ax = fig.add_subplot(111)
bp = ax.boxplot(bxdata)
ax.set_xticklabels(['Best Sol', 'Median Sol', 'Mean Sol'])
plt.show()

fig = plt.figure(1, figsize=(3,6))
ax = fig.add_subplot(111)
bp = ax.boxplot([fitcall])
ax.set_xticklabels(['Fitness Calls'])
plt.show()


# pos = nx.spring_layout(G)
m = -10**10
best = []
for k in range(25):
    if m < fit['best'][k]:
        best = solution[k]

print(best)
for j in range(n):
    if best[j] == 1:
        G.nodes[j]['color'] = "#ff0000"
    else:
        G.nodes[j]['color'] = "#aeaeae"


val = dict()
val['dia'] = []
#val['sigma'] = []
val['clust'] = []
val['comm'] = []

# for k in range(25):
#     print(k)
#     nodeselected = np.where(solution[k] == 1)
#     sg = G.subgraph(nodeselected[0])
#     val['dia'].append(-nx.diameter(sg) / nx.number_of_nodes(sg))
#     #val['sigma'].append(nx.sigma(sg))
#     val['clust'].append(nx.average_clustering(sg))
#
#     # Function of properties of subgraph
#
#     nodelist = sg.nodes()  # ordering of nodes in matrix
#     A = nx.to_numpy_matrix(sg, nodelist)
#     A[A != 0.0] = 1
#     expA = scipy.linalg.expm(A)
#     val['comm'].append(np.mean(expA)/nx.number_of_nodes(sg))
#
#
# print(np.mean(val['dia']))
# #print(np.mean(val['sigma']))
# print(np.mean(val['clust']))
# print(np.mean(val['comm']))
#
#
# print(-nx.diameter(G) / nx.number_of_nodes(G))
# #print(nx.sigma(G))
# print(nx.average_clustering(G))
#
# nodelist = G.nodes()  # ordering of nodes in matrix
# A = nx.to_numpy_matrix(G, nodelist)
# A[A != 0.0] = 1
# expA = scipy.linalg.expm(A)
# print(np.mean(expA)/nx.number_of_nodes(G))


# nx.draw_networkx(G, pos=pos, node_size=10, node_color=list(nx.get_node_attributes(G, 'color').values()),
#                      with_labels=False, width=0.5)
# plt.show()

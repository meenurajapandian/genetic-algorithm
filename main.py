import networkx as nx
import matplotlib.pyplot as plt

import func

G = nx.karate_club_graph()

for u in G.nodes():
    if G.nodes[u]['club'] == 'Mr. Hi':
        G.nodes[u]['color'] = "#ababab"
    else:
        G.nodes[u]['color'] = "#000000"

pos = nx.spring_layout(G)

for u in G.nodes():
    if G.nodes[u]['club'] == 'Mr. Hi':
        G.nodes[u]['color'] = "#ababab"
    else:
        G.nodes[u]['color'] = "#000000"


nx.draw_networkx(G, pos=pos, node_color=nx.get_node_attributes(G,'color').values())


popsize = 4
n = G.number_of_nodes()
result = func.ga(popsize, n, 0.1, 0.8, 100, G)

plt.subplot(2, 2, 1)
nx.draw_networkx(G, pos=pos, node_size=100, node_color=list(nx.get_node_attributes(G, 'color1').values()))

for i in range(popsize):
    for j in range(n):
        if result['population'][i][j] == 1:
            G.nodes[j]['color1'] = "#000000"
        else:
            G.nodes[j]['color1'] = "#ababab"

    plt.subplot(2, 2, (i+1))
    nx.draw_networkx(G, pos=pos, node_size=100, node_color=list(nx.get_node_attributes(G, 'color1').values()))


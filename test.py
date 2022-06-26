from search import *
from takuzu import *
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    startBoards,files = Board.parse_instances_from_dir("/home/jmseca/IST/IA/Projeto/Projeto_IA/testes-takuzu/")
    problems = list(map(Takuzu,startBoards))
    searchers=[breadth_first_tree_search, astar_search, depth_first_graph_search, greedy_search]
    compare_searchers2(problems[:6],["algo"]+[file for file in files][:6])
    print("====================")
    compare_searchers2(problems[6:],["algo"]+[file for file in files][6:])
    """
    searchers_str = ["BFTS","A*","DFTS","Greedy"]
    try:
        problems_results = compare_searchers(problems,searchers=searchers)
    except:
        print("No Solution :(")
        exit(2)
    
    dfs = []
    for problems in problems_results:
        df = pd.DataFrame(columns=["Algorithm","Category","N"])
        for a in range(len(problems)):
            algo_result = problems[a]
            #Gerados
            df.loc[len(df.index)] = [searchers_str[a],"Nós Gerados",algo_result.succs]
            #Expandidos
            df.loc[len(df.index)] = [searchers_str[a],"Nós Expandidos",algo_result.states]
            #Goal Tests
            df.loc[len(df.index)] = [searchers_str[a],"Nós Testados",algo_result.goal_tests]
        dfs += [df]

    for i in range(len(files)):
        sb.barplot(data=dfs[i],x="Category",y='N',hue='Algorithm')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        plt.savefig("Images/"+str(files[i])+".png", bbox_inches='tight')

        plt.clf()
        plt.cla()
        plt.close()
    """



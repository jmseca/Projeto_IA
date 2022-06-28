from search import *
from takuzu import *
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    searchers=[breadth_first_tree_search, astar_search, depth_first_graph_search,greedy_search]
    searchers_str = ["BFTS","A*","DFTS","Greedy"]
    startBoards,files = Board.parse_instances_from_dir("/home/jmseca/IST/IA/Projeto/Projeto_IA/testes-takuzu/")
    problems = list(map(Takuzu,startBoards,[3]*len(startBoards)))
    #compare_searchers2(problems[:6],["algo"]+[file for file in files][:6])
    #print("====================")
    #compare_searchers2(problems[6:],["algo"]+[file for file in files][6:])
    """
        print("=====================")
        print("=====================")
        print("=====================")


    board = Board.parse_instances_from_file("/home/jmseca/IST/IA/Projeto/Projeto_IA/testes-takuzu/input_T13")
    problem1 = Takuzu(board,heur=1)
    problem2 = Takuzu(board,heur=2)
    problems_results = compare_searchers([problem1,problem2],searchers=[astar_search])

    
    """
    
    try:
        problems_results = compare_searchers(problems,searchers=searchers)
    except:
        print("No Solution :(")
        exit(2)
    
    #dfs = []
   
    df = pd.DataFrame(columns=["Size","Free_Pos","Algorithm","Elapsed_Time","Nós Gerados","Nós Expandidos",
        "Nós Testados"])
    
    time_taken_factor = problems_results[0][0].succs/problems_results[0][0].time_taken
    #df = pd.DataFrame(columns=["Category","N","Problema"])
    #i=0
    for problems in problems_results:
        
        #print(len(problems))
        for a in range(len(problems)):
            algo_result = problems[a]
            #Gerados
            df.loc[len(df.index)] = [algo_result.size,
            algo_result.empty_pos,searchers_str[a],algo_result.time_taken,algo_result.succs,
            algo_result.states,algo_result.goal_tests]
            #df.loc[len(df.index)] = ["Tempo",round(algo_result.time_taken*time_taken_factor),
            #"H{}".format(i+1)]
            #df.loc[len(df.index)] = ["Nós Gerados",algo_result.succs,"H{}".format(i+1)]
            #df.loc[len(df.index)] = ["Nós Expandidos",algo_result.states,"H{}".format(i+1)]
        #i+=1
    df.to_excel("excel_data/All.xlsx",index=False)
    """
    print(df)
    #BarPlots
    #for i in range(len(files)):
    #    sb.barplot(data=dfs[i],x="Category",y='N',hue='Algorithm')
    sb.barplot(data=df,x="Category",y='N',hue='Problema')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig("Images/H1VsH2.png", bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()
    
    """
    df2 = df.sort_values(by="Free_Pos")
    data=[[df.loc[i, 'Free_Pos'],
            df.loc[i, 'Elapsed_Time'],
            df.loc[i, 'Algorithm']] for i in np.argsort(df["Free_Pos"].to_list)]
    x = [el[0] for el in data]
    y = [el[1] for el in data]
    C = [el[2] for el in data]
    # Line Plot
    sb.lineplot(data=df2,x="Free_Pos",y="Elapsed_Time",hue="Algorithm")
    plt.savefig("Images/BFS.png", bbox_inches='tight')

    



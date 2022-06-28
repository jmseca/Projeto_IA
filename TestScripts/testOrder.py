from search import *
from takuzu import *
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    searchers=[astar_search, depth_first_graph_search,greedy_search]
    searchers_str = ["A*","DFTS","Greedy"]
    board = Board.parse_instances_from_file("/home/jmseca/IST/IA/Projeto/Projeto_IA/testes-takuzu/input_T09")
    problem1 = Takuzu(board,so=1)
    problem2 = Takuzu(board,so=2)
    problems_results = compare_searchers([problem1,problem2],searchers=searchers)

    
    """
    
    try:
        problems_results = compare_searchers(problems,searchers=searchers)
    except:
        print("No Solution :(")
        exit(2)
    
    #dfs = []
   
    df = pd.DataFrame(columns=["Size","Free_Pos","Algorithm","Elapsed_Time","Nós Gerados","Nós Expandidos",
        "Nós Testados"])
    """
    time_taken_factor = problems_results[0][0].succs/problems_results[0][0].time_taken
    df = pd.DataFrame(columns=["Category","N","Algo"])
    i=0
    for problems in problems_results:
        
        print(len(problems))
        for a in range(len(problems)):
            algo_result = problems[a]
            #Gerados
            #df.loc[len(df.index)] = [algo_result.size,
            #algo_result.empty_pos,searchers_str[a],algo_result.time_taken,algo_result.succs,
            #algo_result.states,algo_result.goal_tests]
            df.loc[len(df.index)] = ["Tempo",round(algo_result.time_taken*time_taken_factor,4),
            searchers_str[a]+"_{}".format(i+1)]
            df.loc[len(df.index)] = ["Nós Gerados",algo_result.succs,searchers_str[a]+"_{}".format(i+1)]
            df.loc[len(df.index)] = ["Nós Expandidos",algo_result.states,searchers_str[a]+"_{}".format(i+1)]
        i+=1
    #df.to_excel("excel_data/All.xlsx",index=False)
    indexNames = df[df['Algo'] == 'A*_2'].index
 
    # Delete these row indexes from dataFrame
    df.drop(indexNames , inplace=True)
    df = df.replace(["A*_1"],'A*')
    print(df)
    #BarPlots
    #for i in range(len(files)):
    #    sb.barplot(data=dfs[i],x="Category",y='N',hue='Algorithm')
    sb.barplot(data=df,x="Category",y='N',hue='Algo')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig("Images/Choice_Test.png", bbox_inches='tight')



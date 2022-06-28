from search import *
from takuzu import *
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    startBoards,files = Board.parse_instances_from_dir("/home/jmseca/IST/IA/Projeto/Projeto_IA/testes-takuzu/")
    #problems1 = list(map(Takuzu,startBoards,[1]*len(startBoards)))
    #problems2 = list(map(Takuzu,startBoards,[2]*len(startBoards)))
    problems3 = list(map(Takuzu,startBoards,[3]*len(startBoards),[1]*len(startBoards)))
    problems4 = list(map(Takuzu,startBoards,[3]*len(startBoards),[2]*len(startBoards)))
    problems_results = compare_searchers(problems=(problems4+problems3),searchers=[astar_search])
    time_taken_factor = problems_results[0][0].succs/problems_results[0][0].time_taken
    df = pd.DataFrame(columns=["Category","N","Problema"])
    i=0
    for problems in problems_results:
        
        print(len(problems))
        for a in range(len(problems)):
            algo_result = problems[a]
            df.loc[len(df.index)] = ["Tempo",round(algo_result.time_taken*time_taken_factor,4),
            "Cost_{}".format(i//len(startBoards)+1)]
            df.loc[len(df.index)] = ["Nós Gerados",algo_result.succs,"Cost_{}".format(i//len(startBoards)+1)]
            df.loc[len(df.index)] = ["Nós Expandidos",algo_result.states,"Cost_{}".format(i//len(startBoards)+1)]
        i+=1

    print(df)
    new_df = pd.DataFrame(columns=["Category","N","Problema"])
    for i in [1,2]:
        tempo_a,ger_a,exp_a = 0,0,0
        df_prob = (df.loc[df['Problema'] == "Cost_{}".format(i)])
        tempos = list(df_prob.loc[df_prob['Category'] == "Tempo"]["N"])
        gers = list(df_prob.loc[df_prob['Category'] == "Nós Gerados"]["N"])
        exps = list(df_prob.loc[df_prob['Category'] == "Nós Expandidos"]["N"])
        tempo_a = sum(tempos)/len(tempos)
        ger_a = sum(gers)/len(gers)
        exp_a = sum(exps)/len(exps)
        new_df.loc[len(new_df.index)] = ["Tempo",tempo_a,"Cost_{}".format(i)]
        new_df.loc[len(new_df.index)] = ["Nós Gerados",ger_a,"Cost_{}".format(i)]
        new_df.loc[len(new_df.index)] = ["Nós Expandidos",exp_a,"Cost_{}".format(i)]


    sb.barplot(data=new_df,x="Category",y='N',hue='Problema')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.savefig("Images/Cost.png", bbox_inches='tight')
    



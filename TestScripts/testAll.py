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
    compare_searchers2(problems[:6],["algo"]+[file for file in files][:6])
    print("====================")
    compare_searchers2(problems[6:],["algo"]+[file for file in files][6:])
    


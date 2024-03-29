# Exemplos de funcoes heuristicas testadas

# 1 
def h(self, node):
    """Função heuristica utilizada para a procura A*."""
    action = node.action
    if action==None or action[0] == 0:      #direct action
        return 0
    else:
        row, col, num = action[1:]
        return - node.state.num_count_row_col(row,col,1-num) + node.state.num_count_row_col(row,col,num)

# 2
def h(self, node):
    """Função heuristica utilizada para a procura A*."""
    action = node.action
    if action==None or action[0] == 0:      #direct action
        return 0
    else:
        row, col, num = action[1:]
        return - node.state.num_count_row_col(row,col,1-num) + node.state.num_count_row_col(row,col,num)\
            - node.state.n_filled

# 3
def H2(self,node):
        action = node.action
        if action==None or action[0] == 0:      #direct action
            return -len(action[1])
        else:
            row, col, num = action[1:]
            adjacent_diff = node.state.num_adjacent_diff(row,col,num)
            return adjacent_diff + abs(adjacent_diff) if adjacent_diff!=0 else num
            
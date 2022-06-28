# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 68:
# 95749 Joao Fonseca
# 95764 Wanghao Zhu

####################################################
# APAGAR ISTO DEPOIS
import os
####################################################

import sys
import numpy as np
from search import (
    InstrumentedProblem,
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
    depth_limited_search
)


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    def __init__(self, size, tabl, order=1):
        self.size = size
        self.tabl = tabl
        self.order = order #apagar

    # apagar
    def set_order(self,so):
        self.order=so
    ##

    def get_number(self, row: int, col: int) -> (int):
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.tabl[row][col]

    def get_row(self, row: int):
        """Devolve todos os valores da linha 'row' do tabuleiro."""
        return self.tabl[row]

    def get_col(self, col: int):
        """Devolve todos os valores da coluna 'col' do tabuleiro."""
        return self.tabl[:,col]

    def put_number(self, row, col, num):
        """Coloca um valor numa posicao do tabuleiro"""
        self.tabl[row][col] = num

    def check_if_row_filled(self, row):
        """Verifica se a linha 'row' está totalmente preenchida."""
        boardRow = self.tabl[row]
        return len(np.bincount(boardRow)) == 2

    def check_if_col_filled(self, col):
        """Verifica se a coluna 'col' está totalmente preenchida."""
        boardCol = self.tabl[:,col]
        return len(np.bincount(boardCol)) == 2

    def adjacent_vertical_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        low = None if (self.size == (row+1)) else self.get_number(row+1, col)
        high = None if (row == 0) else self.get_number(row-1, col)
        return (low,high)

    def adjacent_horizontal_numbers(self, row: int, col: int):
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        right = None if (self.size == (col+1)) else self.get_number(row, col+1)
        left = None if (col == 0) else self.get_number(row, col-1)
        return (left,right)

    def acceptable_zeros_ones_count_row(self, row: int):
        """Verifica se o número de 0's e 1's é válido para uma dada linha"""
        r = self.tabl[row]
        count = np.bincount(r)
        diff = abs(count[0]-count[1])
        return diff==0 or diff==1

    def acceptable_zeros_ones_count_col(self, col: int):
        """Verifica se o número de 0's e 1's é válido para uma dada coluna"""
        col = self.tabl[:,col]
        count = np.bincount(col)
        diff = abs(count[0]-count[1])
        return diff==0 or diff==1
        
    def acceptable_zeros_ones_count(self):
        """Verifica se o número de 0's e 1's é válido"""
        accept = True
        i=0
        while (accept and i<self.size):
            accept = self.acceptable_zeros_ones_count_row(i) and self.acceptable_zeros_ones_count_col(i)
            i+=1
        return accept

    def no_3_adjacent(self):
        """
        Verifica se não há mais do que dois números iguais adjacentes (horizontal ou verticalmente) um ao
        outro
        Devolve True se o tabuleiro respeita a regra, False caso contrario
        """
        accept = True
        limit = (self.size) 
        i=0
        while (accept and i<limit):
            n=0
            while (accept and n<limit):
                elem = self.get_number(i, n)
                hori = self.adjacent_horizontal_numbers(i,n)
                vert = self.adjacent_vertical_numbers(i,n)
                accept = ((hori[0]!=hori[1] or hori[0]!=elem) and (vert[0]!=vert[1] or vert[0]!=elem))
                n+=1
            i+=1
        return accept

    def all_rows_diff(self):
        """Testa se as linhas do tabuleiro são todas diferentes"""
        return len(self.tabl) == len(np.unique(self.tabl,axis=0))

    def all_cols_diff(self):
        """Testa se as colunas do tabuleiro são todas diferentes"""
        return len(self.tabl) == len(np.unique(self.tabl,axis=1))

    def invalidNumberOfOnesZeros(self, row, col, value):
        """
        Verifica numa linha e numa coluna se o número "value" excedeu a quantidade limite 
        """
        limit = np.ceil(self.size/2) #6 -> 3, 7 -> 4
        rowCount = np.bincount(self.tabl[row])[value]
        colCount = np.bincount(self.tabl[:,col])[value]
        return rowCount>limit or colCount>limit


    def get_direct_indirect_pos(self):
        """
        Devolve 2 listas, 1 com as posicoes vazias com valor obrigatorio (a) e outra com as posicoes vazias
        cujo valor ainda não é sabido (b).
        a - [[row,col,val], ... ]
        b - [[1,row,col,val1], [1,row,col,val0]]
        Deteta conflitos. Caso haja, devolve [[],[]]
        """
        direct,indirect=[],[]
        added = [] #lista de pares [row,col] adicionados a direct
        done_indirect = False
        for row in range(self.size):
            for col in range(self.size):
                to_add = []
                n=self.get_number(row,col)
                (left,right)=self.adjacent_horizontal_numbers(row,col)
                (low,high)=self.adjacent_vertical_numbers(row,col)
                if n==2:
                    if left==right and left!=2:
                        to_add = [[row,col,1-left]]
                    elif low==high and low!=2:
                        to_add = [[row,col,1-low]]
                    elif not(done_indirect):
                    #else:
                        indirect = [[1, row,col,0],[1, row,col,1]] if self.order==1 else\
                            [[1, row,col,1],[1, row,col,0]]
                        #indirect += [[row,col]]
                        done_indirect = True
                else:
                    if n==left and right==2:
                        to_add = [[row,col+1,1-n]]
                    elif n==right and left==2:
                        to_add = [[row,col-1,1-n]]
                    if n==low and high==2:
                        to_add += [[row-1,col,1-n]]
                    elif n==high and low==2:
                        to_add += [[row+1,col,1-n]]
                for elem in to_add:
                    if [elem[0],elem[1]] not in added:
                        direct+=[elem]
                        added += [[elem[0],elem[1]]]
                    elif elem[2] != [rcn for rcn in direct if (rcn[0]==elem[0] and rcn[1]==elem[1])][0][2]:
                        return [[],[]]
        #if indirect!=[]:
        #    indirect = [[[1,rc[0],rc[1],0],[1,rc[0],rc[1],1]] for rc in indirect if 
        #        self.num_filled_rc_pos(rc[0],rc[1])==max(list(map(
        #        lambda x : self.num_filled_rc_pos(x[0],x[1]),indirect))
        #    )][0]
        #print(indirect,"ind")
        return [direct,indirect]
                        
    def __str__(self):
        out = ""
        for row in self.tabl:
            for element in row:
                out += str(element)+"\t"
            out = out[:-1] + "\n" 
        return out[:-1]


    def countOccupiedPos(self):
        """Devolve o números de posições já preenchidas no tabuleiro."""
        return np.count_nonzero(self.tabl != 2)

    def countFreePos(self):
        """Devolve o números de posições já preenchidas no tabuleiro."""
        return self.size**2 - np.count_nonzero(self.tabl != 2)

    def num_adjacent_diff(self,row,col,num):
        zeros_ones = [0,0]
        pre_positions = [[row-1,col],[row+1,col],[row,col-1],[row,col+1]]
        positions = list(filter(lambda x:
        x[0]>=0 and x[1]>=0 and x[0]<self.size and x[1]<self.size,pre_positions))
        for pos in positions:
            pos_num = self.get_number(pos[0],pos[1])
            if pos_num is not None and pos_num!=2:
                zeros_ones[pos_num]+=1
        return zeros_ones[(1-num)]-zeros_ones[num]
    
    def num_empty_rc_pos(self,row,col):
        col_bincount = np.bincount(self.get_col(col))
        row_bincount = np.bincount(self.get_row(row))
        col_count = 0 if len(col_bincount)==2 else col_bincount[2]
        row_count = 0 if len(row_bincount)==2 else row_bincount[2]
        return col_count + row_count

    def num_filled_rc_pos(self,row,col):
        col_bincount = np.bincount(self.get_col(col))
        row_bincount = np.bincount(self.get_row(row))
        col_count = col_bincount[0]+col_bincount[1]
        row_count = row_bincount[0]+row_bincount[1]
        return col_count + row_count

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        size = int(sys.stdin.readline())
        tabl = np.zeros(shape=(size,size), dtype=np.ubyte)
        for i in range(size):
            newLine = sys.stdin.readline()
            newLineEl = newLine.split('\t')
            tabl[i] = np.asarray(newLineEl)
        board = Board(size,tabl)
        return board


    # APAGAR ###

    @staticmethod
    def parse_instances_from_dir(dir):
        """
        Lê todos os testes de uma pasta do sistema operativo.
        Retorna uma lista com ok
        """
        boards = []
        files = []
        for file in os.listdir(dir):
            if  file[:6]!='output' and file!='input_T01' and (file[:5]=='input' or file.split('.')[1]=="in"):
                boards += [Board.parse_instances_from_file(file,dir=dir)]
                files += [file]
        return boards,files

    def parse_instances_from_file(file,dir=""):
        """
        Lê todos os testes de uma pasta do sistema operativo.
        Retorna uma lista com ok
        """
        board=None
        try:
            file_open = open(dir+file,'r')
            size = int(file_open.readline())
            tabl = np.zeros(shape=(size,size), dtype=np.ubyte)
            for i in range(size):
                newLine = file_open.readline()
                newLineEl = newLine.split('\t')
                tabl[i] = np.asarray(newLineEl)
            board = Board(size,tabl)
        except:
            print(file)
        return board
    
    def get_empty_pos(self):
        return (self.size**2)-self.countOccupiedPos()

    ##########


    

class TakuzuState:
    state_id = 0
    intConst = 180
    def __init__(self, board: Board, n_filled = -1, rows=[], cols = []):
        #, rows = -1, cols = -1):
        self.board = board
        self.n_filled = n_filled if n_filled!=-1 else board.countOccupiedPos()
        self.id = TakuzuState.state_id
        self.conflicts = False
        self.rows = rows
        self.cols = cols
        self.type = "indirect"
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    def filled(self):
        """Verificar se tabuleiro está totalmente preenchido."""
        return self.n_filled==(self.board.size)*(self.board.size)

    def duplicate(self):
        """Criar um novo TakuzuState."""
        newBoard = Board(self.board.size, np.copy(self.board.tabl))
        newState = TakuzuState(newBoard, self.n_filled, self.rows.copy(), self.cols.copy())
        return newState

    def rowcol_to_number(self, rc):
        """Transformar uma linha ou uma coluna numa lista mais pequena de números."""
        size = self.board.size
        list_bin_intervals = []
        i = 0
        while (size > TakuzuState.intConst):
            list_bin_intervals += [(i+1)*TakuzuState.intConst]
            size -= TakuzuState.intConst
            i+=1
        offset = 0 if i==0 else list_bin_intervals[-1]
        list_bin_intervals += [size+offset]
        size_bin_intervals = i+1
        rc_number = []
        for n in range(size_bin_intervals):
            init = 0 if n==0 else list_bin_intervals[n-1]
            interval = rc[init:list_bin_intervals[n]]
            interval_number = int("".join(str(i) for i in interval),2)
            rc_number += [interval_number]
        return rc_number

    def addNumber(self, row, col, value):
        """
        Adiciona o valor "value" na posicao row,col (linha,coluna) e atualiza o estado
        """
        self.n_filled += 1
        self.board.put_number(row, col, value)
        self.check_for_conflicts(row,col,value)

    def check_valid_new_rowcol(self,row,col):
        """Verificar se existe linhas e colunas iguais a linha 'row' e coluna 'col' respetivamente."""
        if (self.board.check_if_row_filled(row)):
            new_row = self.rowcol_to_number(self.board.get_row(row))
            if new_row not in self.rows:
                self.rows += [new_row]
            else: 
                self.conflicts = True
        if (self.board.check_if_col_filled(col)):
            new_col = self.rowcol_to_number(self.board.get_col(col))
            if new_col not in self.cols:
                self.cols += [new_col]
            else: 
                self.conflicts = True

    def check_for_conflicts(self,row,col,value):
        """Verificar se a linha 'row' e coluna 'col' satisfazem as regras de Takuzu."""
        self.conflicts = self.board.invalidNumberOfOnesZeros(row,col,value)
        if not(self.conflicts):
            self.check_valid_new_rowcol(row,col)

    def num_count_row_col(self,row,col,num):
        """Devolve a soma das quantidades de 'num' na linha 'row' e coluna 'col'."""
        return np.bincount(self.board.get_col(col))[num] + np.bincount(self.board.get_row(row))[num] 

    def num_adjacent_diff(self,row,col,num):
        zeros_ones = [0,0]
        pre_positions = [[row-1,col],[row+1,col],[row,col-1],[row,col+1]]
        positions = list(filter(lambda x:
        x[0]>=0 and x[1]>=0 and x[0]<self.board.size and x[1]<self.board.size,pre_positions))
        for pos in positions:
            pos_num = self.board.get_number(pos[0],pos[1])
            if pos_num is not None and pos_num!=2:
                zeros_ones[pos_num]+=1
        return zeros_ones[(1-num)]-zeros_ones[num]





class Takuzu(Problem):
    # apagar heur pc
    def __init__(self, board: Board, heur=1, pc=1, so=1):
        """O construtor especifica o estado inicial."""
        board.set_order(so)
        super().__init__(TakuzuState(board))
        self.heur = heur   # apagar 
        self.pc = pc # apagar 


    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        #print("=====================")
        #print(state.board)
        if state.conflicts:
            return []
        all_action=state.board.get_direct_indirect_pos()
        if all_action[0]==[]:
            state.type = "direct"
            #print("Indirect")
            #print(all_action[1])
            #print("all_action =",all_action)
            return all_action[1]
        else:
            state.type = "indirect"
            #print("Direct")
            #print(all_action[0])
            return [[0, all_action[0]]]


    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        type = action[0]
        #print("action",action)
        newState = state.duplicate()
        if type == 0:                       #direct action
            for directAction in action[1]:
                if newState.conflicts:
                    break
                row, col, num = directAction
                newState.addNumber(row, col, num)
        else:                               #indirect action
            #print(action[1:])
            row, col, num = action[1:]
            newState.addNumber(row, col, num)
        return newState

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        return state.filled() and not(state.conflicts) #sera que basta isto?
        #return state.filled() and state.board.acceptable_zeros_ones_count() and state.board.no_3_adjacent() \
            #and state.board.all_rows_diff() and state.board.all_cols_diff()

    def H1(self, node):
        action = node.action
        if action==None:            #conflicts
            return np.inf
        elif action[0] == 0:        #direct action
            return -np.inf
        else:
            row, col, num = action[1:]
            return (node.state.num_count_row_col(row,col,num)/((2*node.state.board.size+1)-1))

    def H2(self,node):
        action = node.action
        if action==None:
            return np.inf
        elif action[0] == 0:      #direct action
            return -np.inf
        else:
            row, col, num = action[1:]
            adjacent_diff = node.state.num_adjacent_diff(row,col,num)
            return adjacent_diff + abs(adjacent_diff) if adjacent_diff!=0 else num
            

    def H3(self,node):
        action = node.action
        if action==None:            #conflicts
            return np.inf
        elif action[0] == 0:        #direct action
            return -np.inf
        else:
            row, col, num = action[1:]
            pre_h = node.state.num_count_row_col(row,col,num) - node.state.num_count_row_col(row,col,1-num) -1
            freePos = node.state.board.countFreePos()
            return  freePos-1 if pre_h<=0 else freePos
            #pre_h = node.state.num_count_row_col(row,col,num) - node.state.num_count_row_col(row,col,1-num)
            #return  pre_h 
        
    def H4(self,node):
        action = node.action
        if action==None:            #conflicts
            return np.inf
        elif action[0] == 0:        #direct action
            return node.state.board.countFreePos()
        else:
            row, col, num = action[1:]
            pre_h = node.state.num_count_row_col(row,col,num) - node.state.num_count_row_col(row,col,1-num) -1
            freePos = node.state.board.countFreePos()
            return  freePos-1 if pre_h<=0 else freePos

    
    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        if self.heur==1:
            return self.H1(node)
        elif self.heur==2:
            return self.H2(node)
        elif self.heur==3:
            return self.H3(node)
        else:
            return self.H4(node)

    def pc_old(self,B):
        return -(B.n_filled)

    def pc_new(self,cost_so_far,A,B):
        if A.type == "indirect":
            return cost_so_far+2
        else:
            return cost_so_far+(B.board.countOccupiedPos()-A.board.countOccupiedPos())


    def path_cost(self, cost_so_far, A, action, B):
        if self.pc==1:
            return self.pc_new(cost_so_far,A,B)
        else:
            return self.pc_old(B)
        

    # TODO: outros metodos da classe


if __name__ == "__main__":
    startBoard = Board.parse_instance_from_stdin()
    TakuzuProblem = Takuzu(startBoard,heur=3)
    intru = InstrumentedProblem(TakuzuProblem)
    #finalState = breadth_first_tree_search(TakuzuProblem)
    #finalState = greedy_search(TakuzuProblem)
    #finalState = depth_first_tree_search(InstrumentedProblem(TakuzuProblem))
    #finalState = astar_search(TakuzuProblem)
    #astar_search(intru)
    depth_first_tree_search(intru)
    #try:
    #print(intru.problem.state.board)
    print(intru)
    #except:
    #    print("No Solution :(")
    #    exit(2)
    
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    # pass

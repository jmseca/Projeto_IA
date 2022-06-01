# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 68:
# 95749 Joao Fonseca
# 95764 Wanghao Zhu

import sys
import numpy as np
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe

class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    def __init__(self, size, tabl):
        self.size = size
        self.tabl = tabl
        self.n_filled=0

    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        return self.tabl[row][col]
        

    def adjacent_vertical_numbers(self, row: int, col: int) -> (int, int):
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""
        low = None if (self.size == (row+1)) else self.get_number(row+1, col)
        high = None if (row == 0) else self.get_number(row-1, col)
        return (low,high)

    def adjacent_horizontal_numbers(self, row: int, col: int) -> (int, int):
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
        c = self.tabl[:,c]
        count = np.bincount(c)
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
        limit = (self.size -1) 
        i=1
        while (accept and i<limit):
            n=1
            while (accept and n<limit):
                elem = self.get_number(i, n)
                hori = self.adjacent_horizontal_numbers(i,n)
                vert = self.adjacent_vertical_numbers(i,n)
                accept = ((hori[0]!=hori[1] or hori[0]!=elem) or (vert[0]!=vert[1] or vert[0]!=elem))
                n+=1
            i+=1
        return accept

    def all_rows_diff(self):
        """Testa se as linhas do tabuleiro são todas diferentes"""
        return len(self.tabl) == len(np.unique(self.tabl,axis=0))

    def all_cols_diff(self):
        """Testa se as colunas do tabuleiro são todas diferentes"""
        return len(self.tabl) == len(np.unique(self.tabl,axis=1))
    
    def filled(self):
        return self.n_filled==self.size*self.size

    def get_direct_indirect_pos(self):
        """
        Devolve 2 listas, 1 com as posicoes vazias com valor obrigatorio (a) e outra com as posicoes vazias
        cujo valor ainda não é sabido (b).
        a - [[row,col,val], ... ]
        b - [[1,row,col,val1], [1,row,col,val0]]
        """
        direct,indirect=[],[]
        done_indirect = True
        for row in range(self.size):
            for col in range(self.size):
                n=self.get_number(row,col)
                (left,right)=self.adjacent_horizontal_numbers(row,col)
                (low,high)=self.adjacent_vertical_numbers(row,col)
                if n==2:
                    if left==right and left!=2:
                        direct += [[row,col,1-left]]
                    elif low==high and low!=2:
                        direct += [[row,col,1-low]]
                    elif done_indirect:
                        indirect = [[1, row,col,0],[1, row,col,1]]
                        done_indirect = False
                else:
                    if n==left and right==2:
                        direct += [[row,col+1,1-n]]
                    elif n==right and left==2:
                        direct += [[row,col-1,1-n]]
                    if n==low and high==2:
                        direct += [[row-1,col,1-n]]
                    elif n==high and low==2:
                        direct += [[row+1,col,1-n]]
        return [direct,indirect]
                        




    def __str__(self):
        out = ""
        for row in self.tabl:
            for element in row:
                out += str(element)+"\t"
            out += "\n"
        return out


    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.
        """
        readInput = True
        size = int(sys.stdin.readline())
        tabl = np.zeros(shape=(size,size), dtype=np.ubyte)
        for i in range(size):
            newLine = sys.stdin.readline()
            newLineEl = newLine.split('\t')
            tabl[i] = np.asarray(newLineEl)
        board = Board(size,tabl)
        return board

        

    # TODO: outros metodos da classe


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        all_action=state.board.get_empty_pos_with_certain_value()
        if all_action[0]==[]:
            return all_action[1]
        else:
            return [0, all_action[0]]
        
        # TODO
        pass

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        return self.board.filled() and self.board.acceptable_zeros_ones_count()  and self.board.no_3_adjacent() \
            and self.board.all_rows_diff() and self.board.all_cols_diff()

    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    startBoard = Board.parse_instance_from_stdin()
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.
    pass

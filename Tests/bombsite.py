import sys
import numpy as np

def parse_instance_from_stdin():
    """Lê o test do standard input (stdin) que é passado como argumento
    e retorna uma instância da classe Board.
    """
    readInput = True
    size = int(sys.stdin.readline())
    tabl = np.zeros(shape=(size,size),dtype=np.ubyte)
    for i in range(size):
        newLine = sys.stdin.readline()
        newLineEl = newLine.split('\t')
        tabl[i] = np.asarray(newLineEl)
    print(tabl)


parse_instance_from_stdin()
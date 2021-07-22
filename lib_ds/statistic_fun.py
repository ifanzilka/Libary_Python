##### LINEAL ALGEBRA #####

from typing import List
Vector = List[float]

# OTHER COLOR FOR CONSLOLE
RED = "\033[31m"
FONE = "\033[37m"
GREEN = "\033[32m"

class St_Error (Exception):
    def __init__(self, text):
        tmp = RED + text + FONE
        self.txt = tmp
    def __str__(self):
        return self.txt

def add(v:Vector, w: Vector)->Vector:
    
    #ENG
    """
    Данна функция выполняет сложение двух векторов      

    """
    
    #RUS
    """
    This function perfomans the addition of two vector

    """
    if len(v) != len(w):
        raise St_Error("Error len(v) != len(w) in fun add()!")
    return [v_i + w_i for v_i, w_i in zip(v,w)]

def subtract(v:Vector, w: Vector) -> Vector:
    
    #ENG
    """
    Данна функция выполняет  вычитание двух векторов

    """

    #RUS
    """
    This function perfomans the subtrcat of two vector

    """
    if (len(v) != len(w)):
        raise St_Error("Error len(v) != len(w) in fun subtract()!")
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def vector_sum(vectors:List[Vector])->Vector:
    
    #RUS
    """
    Создаем вектор с покомпонентной суммой элементов всех векторов

    """
    if (len(vectors) < 1):
        raise St_Error("Vectors is empty!")
    num_el = len(vectors[0])
    return [sum(vector[i] for vector in vectors) for i in range(num_el)]

def scalar_multiply(c: float, v: Vector) -> Vector:
    
    #RUS
    """ 
    Умножает каждый элемент на с 
    """
    
    return [c * v_i for v_i in v]

def vector_mean(vectors: List[Vector]) -> Vector:
  #RUS
  """
  Вычисляет поэлементное среднее арифметическое
  """
  n = len(vectors)
  return scalar_multiply(1 / n, vector_sum(vectors))



def dot(v: Vector, w: Vector) -> float:
    
    """Вычисляет v 1 * w 1 + ... + v n * w  или же скалярное произведение """  
    
    if (len(v) != len(w)):
        raise St_Error("The vectros must have the same len")
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

def sum_of_squares(v: Vector) -> float:
  """
  Возвращает v 1 * v 1 + ... + v n * v n
  """
  return dot(v, v)

import math

def magnitude(v: Vector) -> float:
    """Возвращает магнитуду (или длину) вектора v
        math.sqrt - это функция квадратного корня
    """
    return math.sqrt(sum_of_squares(v))

def squared_distance(v: Vector, w: Vector) -> float:
    """Вычисляет (v_l - w_l) ** 2 + ... + (v_n - w_n) ** 2"""
    return sum_of_squares(subtract(v, w)) 


def distance(v: Vector, w: Vector) -> float:
    """Вычисляет расстояние между v и w"""
    return math.sqrt(squared_distance(v, w))


## Mini main from test ##

try:
    print(GREEN,end = '')
    
    
    #print(subtract([5,7,9],[5,6]))
    #print(vector_sum([[1,2],[3,4],[5,6],[7,8]]))
    #print(scalar_multiply(2, [1, 2, 3]))
    #print(dot([1,2,3],[4,5,6]))
    #print(distance([0,0],[1,1]))

    
    print(FONE,end = '')
except Exception as e:
    print(e)

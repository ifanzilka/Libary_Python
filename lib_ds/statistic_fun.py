##### LINEAL ALGEBRA #####

from typing import List
from collections import Counter
from typing import Tuple
from typing import Callable
from typing import TypeVar, List, Iterator 
import random

import math
#import numpy as np

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


### Statistic ###


def mean(xs:List[float])->float:
	"""
	Среднее арифметическое

	Args:
		xs (List[float]): [array numbers]

	Returns:
		float: [mean value]
	"""
	return sum(xs) / len(xs)


def _median_odd(xs: List[float]) -> float:
  """
  Если len(xs) является нечетной,
  то медиана - это срединный элемент
  """
  return sorted(xs) [len(xs) // 2]


def _median_even(xs: List[float]) -> float:
    """
	Если len(xs) является четной, то она является средним значением
	двух срединных элементов
	"""
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2

def median(v: List[float]) -> float:
	"""
		Возвращает элемент находящий по середине списка
	Args:
		v (List[float]): [какой то массив чисел]

	Returns:
		float: Элемент половина которых меньше и половина которых больше
	"""
	
	return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)


def quantile(xs: List[float], p:float) ->float:
  
  """
  Возвращает значение р-го процентиля в х
  """
  p_index = int(p * len(xs))
  return sorted(xs)[p_index] 



def mode(x: List[float]) -> List[float]:
	"""
	Мода - значение или значения, которые встречаются наиболее часто:
	Возвращает список, т. к. может быть более одной моды
	"""
	counts = Counter(x)
	max_count = max(counts.values())
	return [x_i for x_i, count in counts.items() if count == max_count]

def data_range(xs: List[float]) -> float:
	"""
	Разница между макс и мин
	"""
	return max(xs) - min(xs)

def de_mean(xs: List[float]) -> List[float]:
  """
  Транслировать xs путем вычитания его среднего
  (результат имеет нулевое среднее)
   ### X(i) = (x(i)) - M. 
  """
  x_bar = mean(xs)
  return [x - x_bar for x in xs] 

def variance(xs: List[float]) -> float:
  """
  Почти среднеквадратическое отклонение от среднего
  Стандартное отклонение - это корень квадратный из дисперсии
  """
  print('1')
  if (len(xs) < 2):
	  raise St_Error("дисперсия требует наличия не менее двух элементов")
  n = len(xs)
  deviations = de_mean(xs)
  deviations = [x * x for x in deviations]
  return math.sqrt(sum(deviations) / (n - 1))

def interquartile_range(xs: List[float]) -> float:
	"""
	Возвращает разницу между 75%-ным и 25%-ным квартилями
	"""
	return quantile(xs, 0.75) - quantile(xs, 0.25)

def covariance(xs:List[float], ys:List[float])->float:
	"""
	В отличие от дисперсии, которая измеряет отклонение одной-единственной переменной от ее среднего, ковариация измеряет отклонение двух переменных в тандеме от своих средних
	"""
	if (len(xs) != len(ys)):
		raise St_Error("xs и ys должны иметь одинаковое число элементов")
	return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


def correlation(xs: List[float], ys: List[float]) -> float:
	"""
	Измеряет степень, с которой xs и ys варьируются  в тандеме вокруг своих средних
	"""
	stdev_x = variance(xs)
	stdev_y = variance(ys)
	if stdev_x > 0 and stdev_y > 0:
		return covariance(xs, ys) / stdev_x / stdev_y
	else:
		return 0 # если вариации нет, то корреляция равна

#### Probability Theory ####

def uniform_cdf(x: float) -> float:
	"""
	Возвращает вероятность, что равномерно
	распределенная случайная величина <= х
	"""
	if x < 0:
		return 0# Равномерная величина никогда не бывает меньше О
	elif x < 1:
		return x # Например, Р(Х <= 0.4) = 0.4
	else:
		return 1 # Равномерная величина всегда меньше 1


SQRT_TWO_PI = math.sqrt(2 * math.pi)
def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
	"""
	НормаТ/ьное распределение - это классическое колоколообразное распределение.
	Оно полностью определяется двумя параметрами: его средним значением µ (мю)
	и его стандартным отклонением cr (сигмой). Среднее значение указывает, где колокол центрирован, а стандартное отклонение - насколько "широким" он является.
	"""
	return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (SQRT_TWO_PI * sigma))



def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
	"""
	Кумулятивную функцию (CDF) для нормального распределения невозможно написать, пользуясь лишь "элементарными" средствами, однако это можно сделать при
	помощи функции интеграла вероятности math. erf языка Python:
	"""
	return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2 


def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
	"""
	Возвращает mu и sigma, соответствующие Ыnomial (n, р)
	"""
	mu = p * n
	sigma = math.sqrt(p * (1 - p) * n)
	return mu, sigma


# Всякий раз, когда случайная величина подчиняется нормальному распределению,
# мы можем использовать функцию normal _ cdf для выявления вероятности, что ее
# реализованное значение лежит в пределах или за пределами определенного интервала:

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
	"""
	Нормальная функция CDF (normal_cdf) - это вероятность,
	что переменная лежит ниже порога
	Вероятность для нормального распределения P (X <= x)
	"""
	return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

normal_probablity_below = normal_cdf

def normal_probablity_above(lo: float, mu: float = 0, sigma: float = 1) -> float:
	"""
	Вероятность P (X > x) для нормального распределения
	"""
	return 1 - normal_cdf(lo, mu, sigma)


def normal_probablity_between(lo: float, hi: float, mu: float = 0, sigma: float = 1) -> float:
	"""
	Вероятность  P (a < x < b)
	"""
	return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)


def normal_probablity_outside(lo: float,hi: float,mu: float = 0, sigma: float = 1) -> float:
	"""
	1 - P(a < x < b)
	"""
	return 1 - normal_probablity_between(lo, hi, mu, sigma)

# Находим такую точку в которой можем попасть с нужной вероятностью то есть для которой верно P(X <= x)
def inverse_normal_cdf(p: float, mu: float = 0, sigma: float = 1, tolerance: float = 0.0001) -> float: # задать точность
      #"""Отыскать приближенную инверсию, используя бинарный поиск""" 
      # Если не стандарная, то вычислить стандартную и перешкалировать
      if mu != 0 or sigma != 1:
        return mu + sigma * inverse_normal_cdf(p, tolerance = tolerance)
      low_z = -10.0     # normal_cdf(-10) равно (находится очень близко к) О
      hi_z = 10.0       # normal_cdf(l0) равно (находится очень близко к) 1  
      while hi_z - low_z > tolerance:
        mid_z = (low_z + hi_z) / 2  # Рассмотреть среднюю точку
        mid_p = normal_cdf(mid_z)   # и значение CDF
        if mid_p < p:
          low_z = mid_z # Средняя точка слишком низкая, искать выше
        else:
          hi_z = mid_z
      return mid_z


def normal_upper_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
	"""
	Возвращает z, дпя которой P(Z <= z) = вероятность
	"""
	return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float, mu: float = 0, sigma: float = 1) -> float:
	"""
	Возвращает z, дпя которой P(Z >= z) = вероятность
	"""
	return inverse_normal_cdf(1 - probability, mu, sigma)


# Двусторонняя граница
def normal_two_sided_bounds(probability: float, mu: float = 0, sigma: float = 1) -> Tuple[float, float]:
  #"""Возвращает симметрические (вокруг среднего) границы,
  #которые содержат указанную вероятность
  tail_probability = (1 - probability) / 2
  # Верхняя граница должна иметь хвостовую tail_probability вЬШiе ее
  upper_bound = normal_lower_bound(tail_probability, mu, sigma)
  
  # Нижняя граница должгна иметь хвостовую tail_probability ниже ее
  lower_bound = normal_upper_bound(tail_probability, mu, sigma)
  return lower_bound, upper_bound


# Двустороннее р-значение
def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
  # Насколько правдоподобно увидеть значение, как минимум, такое же
  # предельное, что их (в любом направлении), если наши значения
  # поступают из N(mu, sigma)?
  if x >= mu:
    # х больше, чем среднее, поэтому хвост везде больше, чем х
    return 2 * normal_probablity_above(x, mu, sigma)
  else:
    # х меньше, чем среднее, поэтому хвост везде меньше, чем х
    return 2 * normal_probablity_below(x, mu, sigma)


################
### Gradient ###
################


def sum_of_squares(v:List)->float:
  """
  Вычисляет сумму возведенных в квадрат элементов в v
  """
  return dot(v, v)





def difference_quotient(f:Callable[[float], float], x: float, h:float)->float:
	"""
	Проиводная f в точке x , с приближением h 
	если чсило положительное то функция возрастает
	"""
	return (f(x + h) - f(x)) / h

# Частное разностное отношение
def partial_difference_quotient(f: Callable[[List], float], v: List, i: int, h: float) -> float:
	"""Возвращает i-e частное разностное отношение функции f в v"""
	w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
	""" f(2 + 0.001, 3) - f(2,3) / 0.001 если число положительное то возрастаем в этом направлении"""
	return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[List], float],v: List, h: float = 0.0001)->Vector:
	"""
	Возвращаем вектор частных производных
	"""
	return [partial_difference_quotient(f, v, i, h) for i in range(len(v))]


def gradient_step(v: List, gradient: List, step_size: float) -> List:
	"""Движется с шагом 'step_size· в направлении градиента 'gradient' от 'v"""
	if (len(v) != len(gradient)):
		return St_Error("In fun gradient_step len(v) != len(gradien)!")
	step = [i * step_size for i in gradient]
	return [v[i] + step[i] for i in range(len(v))]


def linear_gradient(x: float, y: float, theta: List) -> List:
	slope, intercept = theta #x  и y
	predicted = slope * x + intercept # Модельное предсказание
	error = (predicted - y) # Опмбка равна (предсказание - факт)
	squared_error = error ** 2 # Мы минимизируем квадрат ошибки,
	grad = [2 * error * x, 2 * error] # используя ее градиент
	return grad 


"""
for epoch in range(1000):
  # Вычислить средне значение градиентов
  grad = [linear_gradient(x, y, theta) for x, y in inputs]
  grad_x = [x for x,y in grad]
  grad_y = [y for x,y in grad]
  grad = [np.mean(grad_x),np.mean(grad_y )]
  theta = gradient_step(theta, grad, -learning_rate)     # Сделать шаг в этом направлении
"""

T = TypeVar('Т') # Это позволяет типизировать "обобщенные" функции

def minibatches(dataset: List[T], batch_size: int,shuffle: bool = True) -> Iterator[List[T]]:
	#"""Генерирует мини-пакеты в размере ·ьatch_size' из набора данных"""
	# # start индексируется с О, batch_size, 2 * batch_size, ...
	batch_starts = [start for start in range(0, len(dataset), batch_size)]
	if shuffle:
		random.shuffle(batch_starts) # Перетасовать пакеты
	for start in batch_starts:
		end = start + batch_size
		yield dataset[start:end]

"""
Пример
for epoch in range(500):
  for batch in minibatches(inputs, batch_size=20):
    grad = [linear_gradient(x, y, theta) for x, y in batch]
    grad_x = [x for x,y in grad]
    grad_y = [y for x,y in grad]
    grad = [np.mean(grad_x),np.mean(grad_y )]
    theta = gradient_step(theta, grad, -learning_rate)
"""


# Стохастический градиентный спуск

# for epoch in range(10):
#   for x, y in inputs:
#     grad = linear_gradient(x, y, theta)
#     theta = gradient_step(theta, grad, -learning_rate)
# slope, intercept = theta




# ## Mini main from test ##
# try:

# 	print(GREEN, end = '')
#     #print(subtract([5,7,9],[5,6]))
#     #print(vector_sum([[1,2],[3,4],[5,6],[7,8]]))
#     #print(scalar_multiply(2, [1, 2, 3]))
#     #print(dot([1,2,3],[4,5,6]))
#     #print(distance([0,0],[1,1]))
# 	#print("Ковариация")  
# 	#print(covariance([3,2,2], [3,2,1]))
# 	#print("Кореляция:")
# 	#print(correlation([3, 8, 42],[6, 16, 84])) #0.99
	
# 	#test gradient
# 	# learning_rate = 0.001
# 	# inputs = [ (x, 23 * x + 13) for x in range (-50, 50)]
# 	# theta = [random.uniform(-1, 1), random.uniform(-1, 1)]

# 	# print("Theta")
# 	# print(theta)

# 	# for epoch in range(500):
# 	# 	for batch in minibatches(inputs, batch_size=20):
# 	# 		grad = [linear_gradient(x, y, theta) for x, y in batch]
# 	# 		grad_x = [x for x,y in grad]
# 	# 		grad_y = [y for x,y in grad]
# 	# 		grad = [mean(grad_x),mean(grad_y )]
# 	# 		theta = gradient_step(theta, grad, -learning_rate)

# 	# slope, intercept = theta

# 	# print("Kф при x")
# 	# print(slope)
# 	# print("Кф при y")
# 	# print(intercept)






# 	print(FONE,end = '')
# except Exception as e:
# 	print(e)

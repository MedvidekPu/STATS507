# STATS507

## Prev Assignments

###  Fibonacci sequence
```
def fib():
    yield 1
    yield 1
    x = [1, 1]
    while True:
        a = sum(x)
        yield a
        x = [x[1], a]
```

### Column average
```
def col_avg(csv):
    def mean(v):
        return sum(v) / len(v)
    cols = zip(*[map(float, line.strip().split(",")) for line in csv.split("\n")])
    return list(map(mean, cols))
```

### MyColumn average
```
s=s.split('\n')

p = []

for line in s:
    line = line.split(',')
    for i in range(len(line)):
        line[i] = int(line[i])
    p.append(sum(line)/len(line))
```

### Networking
```
def mapper(line):
    return line.strip().split(" ")

def reducer(accum, x):
    a, b = x
    accum.setdefault((a, b), 0)
    accum[a, b] += 1
    return accum

>>> functools.reduce(reducer, map(mapper, likes.split("\n")), {})
{('A', 'B'): 2, ('C', 'D'): 1, ('B', 'A'): 1, ('A', 'C'): 1, ('C', 'A'): 1}
```

### Trimmed mean
```
def trimmed_mean(x, p):
    k = int(p * len(x))
    return np.mean(x[k:-k])
```

### Bigrams
```
def bigrams(s):
    s = s.strip().lower()
    pairs = map("".join, zip(s[:-1], s[1:]))
    alpha = set(string.ascii_lowercase)
    return collections.Counter([p for p in pairs if set(p) <= alpha])
```

### MyBigrams

```
def bigrams(sen):
    sen = sen.lower().replace(' a ','').replace(' ','').strip()
    
    bi = []
    
    le = len(sen)
    i = 0
    
    for x in range(le):
        bi.append(sen[i:x+2])
        i+=1
        
    for x in bi:
        if len(x)<2:
            bi.remove(x)
        if len(x)>1 and x[0]==x[1]:
            bi.remove(x)
        
    return Counter(bi)
```

### Euclid's algorithm
```
def gcd(a,b):
    while b != 0:
       t = b 
       b = a % b 
       a = t 
    return a
```

### Euler's number appx
```
def euler_limit(n):
    e = (1 + 1/n)**n
    return e
```

```
def euler_infinite_sum(n):
    if n == 0:
        return 0
    else:
        factorial = 1
        euler = 1
        for i in range(1,n):
            factorial = factorial * i
            euler = euler + 1/factorial
        return euler
```
        
```
def euler_approx(epsilon):
    error = euler_infinite_sum(n) - math.exp(1)
    while error < epsilon:
        return euler
```      
      
```        
def print_euler_sum_table(n):
   for i in range(1,n+1):
       print(euler_infinite_sum(i)) 
```

### Palindrome

```
def is_palindrome(s):
    s = s.lower()
    i = 0
    j = len(s)-1
    while i < len(s) and j < len(s):
        if s[i] == " ":
            i += 1
        if s[j] == " ":
            j -= 1
        if s[i] != s[j]:
            return False
        i += 1
        j -= 1
    return True    
```

### Valid sparse vector

```
def is_valid_sparse_vector(s):
    for i in s.keys():
        if not (type(i) == int and i > 0 and type(s[i]) == float):
            return False
    return True
```

### Rotate tuple

```
def rotate_tuple(t, s):
    if type(t) != tuple:
        raise TypeError('input not a tuple')
    if type(s) != int:
        print('input is a non-integer')
        s = int(s)
    first = t[-s:]
    second = t[:-s]
    return first + second
```

### Fibbonaci in class

```
class Fibo:
    def __init__(self):
        self.a = -1
        self.b = 1
        
    def __iter__(self):
        return self
        
    def __next__(self):
        fibo = self.a + self.b
        self.a, self.b = self.b, (self.a + self.b)
        return fibo
```

### Ulam numbers
```
def ulam():
    t = [1,2]
    u = 3
    while True:
        c = 0
        for j in range(0, len(t)): 
            if (u - t[j]) in t and u != (2*t[j]):  
                c += 1
        if c == 2:   
            t.append(u)
            yield u
        u += 1
```
 
 
### Sum of first 10 even square numbers
```
sum_of_even_squares = functools.reduce(lambda x,y: x+y, filter(lambda x: x%2 == 0, [x*x for x in range(1,21)]))
```
 
```
 product_of_primes = functools.reduce(lambda x,y: x*y, [next(p) for i in range(13)])
 # p = primes() generator object
 
 def primes():
    n = 2
    while True:
        if any(n%x==0 for x in range(2,n)) == False:
            yield n
        n+=1
```

### 20 harmonic numbers
```
harmonics = [functools.reduce(lambda x,y: x+1/y, range(i)) for i in range(2,22)]
```

``` 
### Geometric mean of the first tetrahedral numbers
tetra = (n*(n+1)*(n+2)/6 for n in positive())
tetra_geom = [functools.reduce(lambda x,y: x*y, [next(tetra) for x in range(12)])][0]**(1/12)
``` 

### Make polynomials

```
def make_poly(coeffs):
    s = 0
    return lambda x: sum(i * (x ** k) for k,i in enumerate(coeffs))
    
make_poly([1,2,2,3])(2)
# 37
```    
 
### Eval polynomials

``` 
coeffs = [1,2,2,3]
args = [1,2,3]
eval_poly(coeffs, args)
# [8, 37, 106]

def eval_poly(coeffs, args):
    return [make_poly(coeffs)(i) for i in args]
``` 

### Wigner density

``` 
def wigner_density(x):
    if x >= 2 or x <= 2:
        return np.sqrt(4-x**2)/(2*np.pi)
    else:
        return 0
``` 

### Wigner matrix
``` 
def generate_wigner(n):
    if type(n) != int:
        raise TypeError
    if n <= 0:
        raise ValueError
    else:
        x = np.random.normal(0, np.sqrt(1/n), (n,n))
        lower = np.tril(x, k = -1)
        diag = np.diag(np.diagonal(x))
        upper = np.transpose(lower)
        wigner = lower + diag + upper
        wigner = np.matrix(wigner)
        return wigner
``` 

### Eigenvalues

``` 
def get_spectrum(m):
    eigenvalue, eigenvector = np.linalg.eigh(m)
    return eigenvalue
```


### File opening and counting
``` 
def count_bigrams_in_file(file):
    try:
        with open(file, 'r') as f:
            # read file, replace newlines, change to lower case
            s = f.read().replace('\n','').lower()
            # remove punctuations
            punc = set(string.punctuation)
            s = ''.join(char for char in s if char not in punc)
            # split to list
            s = s.split()
            
            dic = {}
            key = []
            for i in range(len(s)-1):
                key.append((s[i], s[i+1]))
            for i in key:
                if i in dic.keys():
                    dic[i] += 1
                else:
                    dic[i] = 1
            return dic
    except IOError:
        print('file cannot be opened')
    except TypeError:
        print('content not string')
```   
  
```   
s = "Half a league, half a league, Half a league onward, All in the valley of Death Rode the six hundred." 
s = s.lower()
punc = set(string.punctuation)
s = ''.join(char for char in s if char not in punc)
s = s.split()
dic = {}
key = []
for i in range(len(s)-1):
    key.append((s[i], s[i+1]))
for i in key:
    if i in dic.keys():
        dic[i] += 1
    else:
        dic[i] = 1
        
#{('half', 'a'): 3,
#('a', 'league'): 3,
#('league', 'half'): 2,
#('league', 'onward'): 1,
#('onward', 'all'): 1,
#('all', 'in'): 1,
#('in', 'the'): 1,
#('the', 'valley'): 1,
#('valley', 'of'): 1,
#('of', 'death'): 1,
#('death', 'rode'): 1,
#('rode', 'the'): 1,
#('the', 'six'): 1,
#('six', 'hundred'): 1}

``` 
### Ruler sequence using itertools

``` 
from itertools import repeat,count

i   = count(1,1)
rul = (n for r in count(1,1) for _ in range(r) for n in repeat(next(i),r))
``` 

## Notes

### Reduce function

``` 
>>> def do_and_print(t1, t2):
    print 't1 is', t1
    print 't2 is', t2
    return t1+t2

>>> reduce(do_and_print, ((1,2), (3,4), (5,)))
t1 is (1, 2)
t2 is (3, 4)
t1 is (1, 2, 3, 4)
t2 is (5,)
(1, 2, 3, 4, 5)
``` 

### Itertools vs. Vanilla Python

```
iterator = (x**2 for x in range(20))
list(itertools.islice(iterator, 2, 10))
# [4, 9, 16, 25, 36, 49, 64, 81]
```

```
iterator = (x**2 for x in range(20))
[x for i, x in enumerate(iterator) if i>=2 and i<10]
# [4, 9, 16, 25, 36, 49, 64, 81]
```


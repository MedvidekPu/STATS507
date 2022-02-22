# STATS507

## Practice Test

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

### Networking
```
def mapper(line):
    return line.strip().split(" ")

def reducer(accum, x):
    a, b = x
    accum.setdefault((a, b), 0)
    accum[a, b] += 1
    return accum
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



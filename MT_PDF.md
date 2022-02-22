# STATS507

## Practice Test

###  Fibonacci sequence
´´´
def fib():
    yield 1
    yield 1
    x = [1, 1]
    while True:
        a = sum(x)
        yield a
        x = [x[1], a]
´´´

### Column average
´´´
def col_avg(csv):
    def mean(v):
        return sum(v) / len(v)
    cols = zip(*[map(float, line.strip().split(",")) for line in csv.split("\n")])
    return list(map(mean, cols))
´´´

def wrapper(S, coins):
    states = [(10000, set()) for k in range(S+1)]
    states = [(10000, []) for k in range(S+1)]
    return n_coins(S, coins, states)

def n_coins(S, coins, states):
    if S < 1:
        return (10000, [])
    if S in coins:
        return (1, [S])
    if S < min(coins):
        return (10000, [])
    if states[S][0] < 10000:
        return states[S]
    for c in coins:
        print S, states[S]
        if c > S:
            continue
        new_s = S - c
        new_state = n_coins(new_s, coins, states)
        new_state = (new_state[0]+1, new_state[1] + [c])
        if new_state[0] < states[S][0]:
            states[S] = new_state
    return states[S]

def n_coins_iter(S, coins):
    states = [(10000, []) for k in range(S+1)]
    states[0] = (0, [])
    for s in range(1,S+1):
        for c in coins:
            if c <= s and states[s-c][0] < states[s][0]:
                states[s] = (states[s-c][0] + 1, states[s-c][1] + [c])
    print states, states[S]

S = int(raw_input("S: "))
coins = set(int(k) for k in raw_input("Coins(comma seperated): ").split(','))
if min(coins) < 1:
    raise Exception("Coins should be positive values >= 1")
print S, coins
print wrapper(S, coins)
print n_coins_iter(S, coins)

def wrapper(arr):
    states = [1]*len(arr)
    return longest_sub(arr, len(arr)-1, states)

def longest_sub(arr, i, states):
    if i <= 0:
        return 1
    if states[i] > 1:
        return states[i]
    for j in range(i):
        lj = longest_sub(arr, j, states)
        if arr[j] <= arr[i]:
            print j, i, states
            states[i] = lj + 1
        else:
            states[i] = lj
    return states[i]

arr = [int(k) for k in raw_input("Array(comma seperated): ").split(',')]
print arr
print wrapper(arr)

def wrapper(mat, N, M):
    states = [[0 for c in range(M)] for r in range(N)]
    return apples(mat,0,0,N,M, states)


def apples(mat,i,j,N,M, states):
    if i >= N or j >= M:
        return 0
    if states[i][j] > 0:
        return states[i][j]
    states[i][j] = mat[i][j] + max([apples(mat,i+1,j,N,M, states), apples(mat,i,j+1,N,M, states)])
    return states[i][j]

arr = [int(k) for k in raw_input("Array(comma seperated): ").split(',')]
N, M = [int(k) for k in raw_input("N, M (comma seperated): ").split(',')]
mat = [arr[i*N:i*N+M] for i in range(N)]
print mat
wrapper(mat, N, M)




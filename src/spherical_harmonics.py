# code based from https://qiita.com/KionIm/items/315d4b68bab39ad8d786

import math

# Factorial
def Factorial(n: int) -> float:
    nm: int = n
    Out: int = n
    if (nm == 0):
        return 1
    elif (nm == 1):
        return 1
    while (nm > 1):
        nm = nm - 1
        Out *= nm
    return Out

# (8),(7)
def Pnpm(n: int, m: int, ct: float) -> float:
    if (n == 0 and m==0):
        return 1.0
    elif (n == 1 and m == 0):
        return ct
    elif (n == 1 and m == 1):
        return math.sqrt(1 - ct * ct)
    elif (n == 2 and m == 1):
        return 3 * ct * math.sqrt(1 - ct * ct)
    else:
        nm: int = n - 1
        return ((2 * nm + 1) *ct* Pnpm(nm, m, ct) - (nm + m) * Pnpm(nm - 1,m, ct)) /float(nm - m + 1)

# (6)
def Pnn(n: int, ct: float) -> float:
    return Factorial(2 * n) * pow(1 - ct * ct, 0.5 * n) / (pow(2, n) * Factorial(n))

# (5)
def Pnmp(n: int, m: int, ct: float) -> float:
    if (m == 0):
        return Pnpm(n, 0, ct)
    elif (m == 1):
        return Pnpm(n, 1, ct)
    elif (n == 0):
        return Pnpm(0, 0, ct)
    elif (n == m):
        return Pnn(n, ct)
    else:
        mm: int = m - 1
        nm: int = n - 1
        return ct * Pnmp(nm, m, ct) + (nm + mm + 1) * math.sqrt(1 - ct * ct) * Pnmp(nm, mm, ct)

# (3)
def Xnm(n: int, m: int, ct: float) -> float:
    S: float = (2 * n + 1) / (4 * math.acos(-1))*Factorial(n-abs(m))/float(Factorial(n+abs(m)));
    S = math.sqrt(S)

    P: float = Pnmp(n, m, ct)

    if (m >= 0):
        return S * P * pow(-1, m)
    else:
        return S * P

# (4)
def Ynm(n: int, m: int, theta: float, phi: float) -> float:
    ct: float = math.cos(theta)
    if (m < 0):
        return math.sqrt(2) * Xnm(n, -m, ct) * math.cos(m * phi)
    elif(m==0):
        return Xnm(n, m, ct)
    else:
        return math.sqrt(2) * Xnm(n, m, ct) * math.sin(m * phi)

def knm(k: int, MMax: int) -> (int, int):
    l: int = MMax + 1 - 1
    if (k <= l):
        m: int = 0
        n: int = k
        return(n, m)

    for i in range(1, MMax+1):
        lp: int = l
        l += 2 * (MMax+1 - i)
        if (k <= l):
            if (k % 2 == 0):
                m: int = i
            else:
                m: int = -i

            n = int((k - lp + 1) / 2 + abs(m)-1)
            return (n, m)

if __name__ == "__main__":

    NumMM = 200
    MMax = 11

    for i in range(0, NumMM):
        print(i, knm(i, MMax))

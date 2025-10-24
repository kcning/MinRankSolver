from sage.all import *

def generate_solvable_minrank_instance(n, m, k, r, field=GF(16)):
    """
    Generate a solvable MinRank instance:
    Returns list of l matrices M_k (m x n) over 'field' and a solution vector alpha
    such that rank(sum(alpha_k * M_k)) <= r.

    Parameters:
    n, m: matrix dimensions
    k: number of matrices
    r: target max rank (r < min(m,n))
    field: finite field over which matrices are defined

    Returns:
    M_list: list of matrices [M_1, ..., M_l]
    alpha: vector of length l (solution)
    """
    # 1. Random solution vector alpha with last element nonzero
    alpha = vector(field, [field.random_element() for _ in range(k-1)] + [field.random_element(nonzero=True)])

    # 2. Construct a random matrix SUM of rank at most r
    # Construct SUM as S * L * T, S (n x r), L (r x r diag), T (r x m)
    S = random_matrix(field, n, r)
    T = random_matrix(field, r, m)
    # For simplicity, L is the identity matrix r x r (rank exactly r)
    L = identity_matrix(field, r)
    SUM = S * L * T

    M0 = random_matrix(field, n, m)
    SUM -= M0

    # 3. Generate M_1 ... M_{l-1} random
    M_list = [random_matrix(field, n, m) for _ in range(k - 1)]

    # 4. Solve for M_l to satisfy sum alpha_k M_k = SUM
    sum_M = sum(alpha[l] * M_list[l] for l in range(k - 1))
    inv_alpha_l = ~alpha[k - 1]  # multiplicative inverse in field
    Ml = inv_alpha_l * (SUM - sum_M)
    M_list.append(Ml)
    M_list.insert(0, M0)

    return M_list, alpha

# Example of usage
n, m = 5, 9  # dimensions
k = 5        # number of matrices
r = 3        # target max rank
q = 16
F = GF(q)   # field
M_list, alpha = generate_solvable_minrank_instance(n, m, k, r, F)

print(f"n = {n}")
print(f"m = {m}")
print(f"k = {k}")
print(f"r = {r}")
for i, M in enumerate(M_list):
    print(f"M{i}:")
    #print(M)
    if q.is_prime():
        str_per_row = [ f"{' '.join(map(str, [c for c in row]))}" for row in M]
    else:
        str_per_row = [ f"{' '.join(map(str, [c.to_integer() for c in row]))}" for row in M]
    print(f"{'\n'.join(str_per_row)}")
    print('')


# Verify rank of sum alpha_k*M_k <= r
Msum = sum(alpha[l] * M_list[l+1] for l in range(k)) + M_list[0]
print("#Rank of sum:", Msum.rank())
if q.is_prime():
    print("#Solution:", [c for c in alpha])
else:
    print("#Solution:", [c.to_integer() for c in alpha])

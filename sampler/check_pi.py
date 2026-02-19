import torch
from collections import deque

path = "gt_text8_char_T1024_N128_topk27_full_eps0.pt"
gt = torch.load(path, map_location="cpu")

# --- Try to get sparse transitions in (nbr_idx, nbr_prob) format
nbr_idx = None
nbr_prob = None
if isinstance(gt, dict):
    for k in ["nbr_idx", "P_topk_idx", "topk_idx"]:
        if k in gt: nbr_idx = gt[k]
    for k in ["nbr_prob", "P_topk_prob", "topk_prob"]:
        if k in gt: nbr_prob = gt[k]

# --- Or dense P
P = None
if isinstance(gt, dict):
    for k in ["P", "P_dense", "transition", "P_full"]:
        if k in gt: P = gt[k]

# Infer V and build adjacency list
if P is not None:
    P = P.float()
    V = P.shape[0]
    adj = [torch.nonzero(P[i] > 0, as_tuple=False).squeeze(1).tolist() for i in range(V)]
    has_self_loop = any(P[i, i].item() > 0 for i in range(V))
elif nbr_idx is not None:
    nbr_idx = nbr_idx.long()
    V = nbr_idx.shape[0]
    adj = [nbr_idx[i].tolist() for i in range(V)]
    # If probs exist, self-loop means i appears in its neighbor list with prob>0
    if nbr_prob is not None:
        nbr_prob = nbr_prob.float()
        has_self_loop = any(((nbr_idx[i] == i) & (nbr_prob[i] > 0)).any().item() for i in range(V))
    else:
        has_self_loop = any((nbr_idx[i] == i).any().item() for i in range(V))
else:
    raise ValueError("Could not find transitions in the .pt file (dense P or topk neighbors).")

# Build reverse graph for SCC check (Kosaraju)
radj = [[] for _ in range(V)]
for u in range(V):
    for v in adj[u]:
        radj[v].append(u)

# 1) DFS order
seen = [False]*V
order = []
def dfs1(s):
    stack = [(s, 0)]
    seen[s] = True
    while stack:
        u, it = stack[-1]
        if it < len(adj[u]):
            v = adj[u][it]
            stack[-1] = (u, it+1)
            if not seen[v]:
                seen[v] = True
                stack.append((v, 0))
        else:
            order.append(u)
            stack.pop()

for i in range(V):
    if not seen[i]:
        dfs1(i)

# 2) Reverse DFS to get SCCs
comp = [-1]*V
def dfs2(s, cid):
    dq = deque([s])
    comp[s] = cid
    while dq:
        u = dq.popleft()
        for v in radj[u]:
            if comp[v] == -1:
                comp[v] = cid
                dq.append(v)

cid = 0
for u in reversed(order):
    if comp[u] == -1:
        dfs2(u, cid)
        cid += 1

num_scc = cid
largest = max([comp.count(c) for c in range(num_scc)])
print(f"V={V}, num_SCC={num_scc}, largest_SCC_size={largest}, has_self_loop={has_self_loop}")

if num_scc == 1:
    print("=> Irreducible (strongly connected).")
    if has_self_loop:
        print("=> Aperiodic (self-loop exists). Unique stationary distribution + convergence.")
    else:
        print("=> Periodicity unknown (no self-loop found). Still unique stationary distribution, but convergence may be periodic.")
else:
    print("=> Not irreducible. eps=0 may admit multiple stationary distributions.")

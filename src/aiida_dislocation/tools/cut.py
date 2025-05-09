from aiida import orm
from ase import Atoms
from ase.build import make_supercell
import numpy
import math
from fractions import Fraction


def group_by_plane(
    ase_atoms: Atoms,
    P,
    cut_plane: list[float],
    tol: float = 1e-6
    ):

    grouped_by_plane = {}
    
    supercell = make_supercell(ase_atoms, P)

    cut_plane = numpy.dot(cut_plane, ase_atoms.cell)

    c = numpy.linalg.norm(cut_plane)
    
    for atom in supercell:
        z_position = numpy.dot(atom.position, cut_plane) / c
        match = next(
            (k for k in grouped_by_plane.keys() 
            if numpy.isclose(k, z_position, atol=tol)),
            None
        )
        if match is None:
            grouped_by_plane[z_position] = [atom]
        else:
            grouped_by_plane[match].append(atom)

    return grouped_by_plane

def plot_layer(
    ase_atoms: Atoms,
    grouped_layer,
    ax, 
    x,
    y,
    color,
    symbol,
    ):
    
    ax.set_xlabel(f"[{x[0]}{x[1]}{x[2]}] ($\AA$)")
    ax.set_ylabel(f"[{y[0]}{y[1]}{y[2]}] ($\AA$)")
    
    x = numpy.dot(x, ase_atoms.cell)
    y = numpy.dot(y, ase_atoms.cell)
    
    
    xs = []
    ys = []
    for atom in grouped_layer:
        if atom.symbol == symbol:
            positions = atom.position
            xs.append(numpy.dot(positions, x) / numpy.linalg.norm(x))
            ys.append(numpy.dot(positions, y) / numpy.linalg.norm(y))
            
    ax.axvline(x=0, color='black', linewidth=1, linestyle='--')
    ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
    ax.scatter(xs, ys, s=50, color=color)
    
    ax.set_aspect('equal')

def find_plane_pbc(
    ase_atoms: Atoms,
    cut_plane: list[float],
    max_denom: int = 10**6,
    tol: float = 1e-6
) -> list[list[float]]:
    """
    Find the plane that cuts the atoms in the PBC box.
    """
    cell = ase_atoms.cell.array
    

    # 2. Miller 法向量
    n = numpy.dot(cut_plane, cell)
    
    n2 = numpy.dot(cell, n)
    # 3. 浮点→最小整数 (A,B,C)
    fracs = [Fraction(x).limit_denominator(max_denom) for x in n2]
    D = math.lcm(*(f.denominator for f in fracs))
    A, B, C = [int(f * D) for f in fracs]
    g_all = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    A, B, C = A//g_all, B//g_all, C//g_all

    print(A, B, C)
    # 4. 构造初始基
    g = math.gcd(A, B)
    v1 = numpy.array([ B//g, -A//g, 0 ], dtype=int)
    v2 = numpy.cross(numpy.array([A, B, C], int), v1)

    # 内积与范数
    def dot(u,v): return int(u.dot(v))
    def norm2(u): return dot(u,u)

    # 5. Gauss 约减
    while True:
        mu = round(Fraction(dot(v1, v2), norm2(v1)))
        v2 = v2 - mu * v1
        if norm2(v2) < norm2(v1):
            v1, v2 = v2, v1
            continue
        break

    v3 = numpy.cross(v1, v2)                        # w = v1 × v2
    g = math.gcd(math.gcd(abs(v3[0]), abs(v3[1])), abs(v3[2]))
    n = v3 // g       
    return (v1, v2, v3)


def basis_transform(
    v1, v2, v3,
    old_atoms, 
    tol=1e-6
    ):
    """
    v1, v2, v3 : np.array(shape=(3,)), 原三维晶格基矢
    old_atoms  : list of (x,y,z)，原胞内所有原子的分数坐标
    tol        : 判断 gamma 是否为整数的容差

    返回：
      一个列表，元素为 (alpha_mod1, beta_mod1, layer_index, real_xyz)
      layer_index 是 gamma 四舍五入后的整数层号，
      real_xyz 是实空间坐标。
    """
    # 2) 计算 M⁻¹（它恰好也是整数矩阵的逆，但用浮点足够精度）
    M  = numpy.column_stack((v1, v2, v3))
    Minv = numpy.linalg.inv(M)

    results = []
    for x,y,z in old_atoms:
        f = numpy.array([x, y, z], dtype=float)
        alpha, beta, gamma = Minv.dot(f)

        # 3) 筛出落在过原点平面上的那些 atom
        if abs(gamma - round(gamma)) > tol:
            continue

        layer = int(round(gamma))
        # 把 alpha,beta 映射回 [0,1)
        a_mod = alpha - np.floor(alpha)
        b_mod = beta  - np.floor(beta)
        # 4) 真实空间坐标
        R = x*a + y*b + z*c

        results.append((a_mod, b_mod, layer, R))

    return results

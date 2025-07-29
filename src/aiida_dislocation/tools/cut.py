from aiida import orm
from ase import Atoms
from ase.build import make_supercell
import numpy
import sympy
import math
from fractions import Fraction

import warnings
import functools
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from matplotlib import patheffects
from matplotlib.patches import Circle
import string
from pymatgen.transformations.standard_transformations import SupercellTransformation

LIST_ALPHABET = list(string.ascii_uppercase)

def list_to_tex(indices):
    """
    将一个整数列表（如 [1, -1, 0]）转换为 Miller 指数的 LaTeX 格式，
    例如 [1, -1, 0] → '$1\\bar{1}0$'。
    """
    out = []
    for h in indices:
        if h < 0:
            # 负数用 \bar{digit} 表示
            out.append(r"\bar{%d}" % abs(h))
        else:
            # 非负数直接写数字
            out.append(str(h))
    return "[$" + "".join(out) + "$]"


def draw_sphere(ax, x, y, radius, base_color='skyblue', base_alpha=1,
                n_rings=150, inner=0.2, outer=0.4, power=1.):
    """
    在 (x,y) 画一个假 3D 球：
    - fac = i/n_rings 为环带半径比例，从 1 到 0 递减
    - fac >= outer: 纯 base_color
    - inner < fac < outer: t = ((fac-inner)/(outer-inner))**power 渐变
    - fac <= inner: 纯白
    - 最外层再加一圈黑边
    """
    base_rgb = np.array(plt.cm.colors.to_rgb(base_color))
    white    = np.array([1.0, 1.0, 1.0])

    for i in range(n_rings, 0, -1):
        fac = i / n_rings
        r   = radius * fac

        if fac <= inner:
            color = white
        elif fac <= outer:
            # 在 inner<fac<=outer 之间做渐变：t 从 0→1
            t = (fac - inner) / (outer - inner)
            w = t**power
            # 由白→base_color
            color = white * (1 - w) + base_rgb * w
        else:
            color = base_rgb

        circ = Circle(
            (x, y), r,
            facecolor=color,
            edgecolor='none',
            alpha=1.0*base_alpha,
            zorder=10
            )
        ax.add_patch(circ)

    # 最外层黑色边框
    rim = Circle(
        (x, y), radius,
        facecolor='none',
        edgecolor='black',
        linewidth=1.5,
        zorder=11
        )
    ax.add_patch(rim)


def draw_edge_arrows(
    ax, a2, b2, 
    frac_pos=0.2,    # 在边上位置：0.2→20%处
    arrow_frac=0.1,  # 箭头长度：边长的 10%
    offset=0.02):    # 垂直的偏移量（同样是数据坐标）
    # 计算 a 边的箭头
    # 1) 单位方向
    ua = a2 / np.linalg.norm(a2)
    # 2) 箭头实际长度向量
    la = ua * np.linalg.norm(a2) * arrow_frac
    # 3) 在边上的起点
    pa = a2 * frac_pos
    # 4) 垂直向上偏移（顺时针旋转 90°）
    na = np.array([-ua[1], ua[0]])  # 单位法向    
    pa_off = pa + na * offset

    # 画箭头
    ax.arrow(
        pa_off[0], pa_off[1], 
        la[0], la[1],
        head_width=-offset*0.5, 
        head_length=-offset*0.5,
        fc='k', ec='k', linewidth=1.2, zorder=15)

    # 同理，b 边的箭头
    ub = b2 / np.linalg.norm(b2)
    lb = ub * np.linalg.norm(b2) * arrow_frac
    pb = b2 * frac_pos
    # b 边的法向（逆时针旋转 90°）
    nb = np.array([ub[1], -ub[0]])
    pb_off = pb + nb * offset

    ax.arrow(
        pb_off[0], pb_off[1],
        lb[0], lb[1],
        head_width=-offset*0.5,
        head_length=-offset*0.5,
        fc='k', ec='k', linewidth=1.2, zorder=15)

def draw_edge_arrow_with_label(
    ax, vec, miller, *,
    frac_pos=0.2,
    arrow_frac=0.1,
    offset=0.03,
    text_offset=(0.1, 0.1),
    fontsize=12,
    **arrow_kwargs):
    """
    在 ax 上沿向量 vec 画一条小箭头，并在箭头旁标注 miller（字符串）。
    
    vec          : np.array, 2 元素，箭头方向向量
    miller       : str，要标注的 Miller 指数，如 "[100]"
    frac_pos     : float, 箭头起点在 vec 上的比例
    arrow_frac   : float, 箭头长度占 vec 长度的比例
    offset       : float, 垂直偏移量，保证不遮边
    text_offset  : tuple(float,float), 文本相对于箭头尾部的偏移 (dx, dy)
    arrow_kwargs : 传给 ax.arrow 的其它参数
    """
    # 1) 计算单位方向和法向
    u = vec / np.linalg.norm(vec)
    # 垂直偏移：顺时针旋转90°
    n = np.array([-u[1], u[0]])

    # 2) 箭头起点
    start = vec * frac_pos + n * offset
    # 3) 箭头长度向量
    length = np.linalg.norm(vec) * arrow_frac
    delta = u * length
    angle_deg = np.degrees(np.arctan2(delta[1], delta[0]))
    # 4) 画箭头
    ax.arrow(
        start[0], start[1], delta[0], delta[1],
        head_width=-offset * 0.5,
        head_length=-offset * 0.5,
        length_includes_head=True,
        **arrow_kwargs
        )

    # 5) 在箭头尾部放文本
    # tx = start[0] + text_offset[0]
    # ty = start[1] + text_offset[1]
    
    tx = start[0] 
    ty = start[1] 
    ax.text(
        tx, ty, miller,
        fontsize=fontsize,
        color=arrow_kwargs.get("color", "black"),
        va="bottom", ha="left",
        # rotation=angle_deg,
        # rotation_mode="anchor",
        zorder=arrow_kwargs.get("zorder", 15)+1
        )
    
# Constants\ nLIST_ALPHABET = [chr(i) for i in range(ord('A'), ord('Z')+1)]

# Utility to compute 2D projection basis
def compute_projection_basis(lattice, axis=0):
    """
    Given a 3x3 lattice and an axis index (0=a,1=b,2=c),
    return the 2x2 in-plane basis matrix B2D and 2D basis vectors a2,b2.
    """
    a3 = lattice[1]
    b3 = lattice[2]
    la, lb = np.linalg.norm(a3), np.linalg.norm(b3)
    costheta = np.dot(a3, b3) / (la * lb)
    theta    = np.arccos(np.clip(costheta, -1, 1))
    # print(la, lb, costheta, theta)
    # B2D: maps frac coords [f_a,f_b] to Cartesian in-plane coords
    B2D = np.array([[la, 0], [lb*costheta, lb*np.sin(theta)]])
    a2 = B2D[0]
    b2 = B2D[1]
    return B2D, a2, b2
def annotate_miller(ax, v1, v2, label1, label2):
    """
    Plots two basis vectors v1 and v2 from the origin and labels their edges with Miller indices.
    
    Parameters:
    - v1, v2: 2D numpy arrays or lists representing the basis vectors.
    - label1, label2: Strings for the Miller index labels corresponding to v1 and v2.
    """
    
    # Calculate midpoints
    mid1 = 0.5 * np.array(v1) - [0., 0.2]
    mid2 = 0.5 * np.array(v2) - [0.1, 0.]
    
    # Label the midpoints with Miller indices
    ax.text(mid1[0], mid1[1], label1, ha='center', va='bottom', fontsize=20)
    ax.text(mid2[0], mid2[1], label2, ha='right', va='center', fontsize=20)
    
# Function 1: layer grouping
def group_structure_layers(struct: Structure, axis=0, tol=1e-6):
    """
    Partition structure into layers along fractional axis.  
    Return:
        layers      : unique fractional values sorted
        layer_masks : list of boolean masks for each layer
        frac_coords : Nx3 array of fractional coords
    """
    frac_coords = np.array(struct.frac_coords)
    vals        = frac_coords[:, axis]
    rounded     = np.round(vals / tol) * tol
    layers      = np.unique(rounded)
    layers      = np.sort(layers)
    masks = [np.isclose(rounded, lv, atol=tol) for lv in layers]
    return layers, masks, frac_coords

# Function 2: plot a single layer
def plot_layer(ax, pts2d, species, layer_index, a2, b2,
                miller_a, miller_b, alphabet=LIST_ALPHABET):
    """
    Plot atoms of one layer and the parallelogram basis on the given axes:
        pts2d   : Mx2 array of Cartesian 2D coordinates for this layer
        species : M-array of species symbols
        layer_index: integer index of this layer (for labeling)
        a2,b2   : 2D basis vectors
        draw_sphere: function(ax,x,y,rad,color,base_alpha)
        draw_edge_arrow_with_label: function(ax,vec,label,...)
        miller_a, miller_b : lists or strings
    """
    # color map
    unique_species = list(dict.fromkeys(species))
    palette        = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
    color_map      = {sp: palette[i % len(palette)] for i, sp in enumerate(unique_species)}
    radius = min(np.linalg.norm(a2), np.linalg.norm(b2)) * 0.1

    # draw atoms
    for sp in unique_species:
        mask = np.array(species) == sp
        xs, ys = pts2d[mask,0], pts2d[mask,1]
        col = color_map[sp]
        for x,y in zip(xs, ys):
            draw_sphere(ax, x, y, radius, col, base_alpha=1.0)
            # label letter
            letter = alphabet[layer_index]
            ax.annotate(
                sp,
                xy=(x,y), xytext=(radius+2, radius+2), textcoords='offset points',
                fontsize=16, color='black', weight='bold', ha='center', va='center', zorder=10
            )

        
    # draw parallelogram basis
    origin = np.zeros(2)
    quad   = np.vstack([origin, a2, a2+b2, b2, origin])
    ax.plot(quad[:,0], quad[:,1], '--', color='black', lw=2, zorder=1)
    # miller labels on edges
    # draw_edge_arrow_with_label(
    #     ax, a2, list_to_tex(miller_a), frac_pos=0.5, arrow_frac=0.2, offset=-0.3,
    #     text_offset=(0.1, 0.1),
    #     color="black", linewidth=1.5, zorder=15)
    # draw_edge_arrow_with_label(ax, b2, list_to_tex(miller_b), frac_pos=0.5, arrow_frac=0.2, offset=-0.3,
    #     text_offset=(0.1, 0.1),
    #     color="black", linewidth=1.5, zorder=15)
    annotate_miller(ax, a2, b2, list_to_tex(miller_a), list_to_tex(miller_b))
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    
    letter = alphabet[layer_index]
    ax.set_title(f"{letter} layer", fontsize=20)

def get_transformed_cell(
    old_struct, 
    slipping_system,
    unit_rep_inv
    ):
    
    M = np.array(
        [
            np.dot(slipping_system['cut_plane'], unit_rep_inv),
            np.dot(slipping_system['a'], unit_rep_inv),
            np.dot(slipping_system['b'], unit_rep_inv)
        ], 
        dtype=int
        )
    dst = SupercellTransformation(M)
    struct = dst.apply_transformation(old_struct)
    return struct

def plot_layers(struct, axes, slipping_system):
    
    layers, masks, fracs = group_structure_layers(struct, axis=0)

    sites = list(struct)

    if len(layers) != len(axes):
        raise ValueError(f"Number of layers ({len(layers)}) does not match number of axes ({len(ax)})")
    
    B2D, a2, b2 = compute_projection_basis(struct.lattice.matrix, axis=0)

    for i, mask in enumerate(masks):
        # 1) 投影坐标
        pts2d = fracs[mask][:, [1,2]].dot(B2D)
        layer_sites = [site for site, m in zip(sites, mask) if m]
        species     = [site.species_string for site in layer_sites]
        # 3) 绘这一层
        plot_layer(
            axes[i], pts2d, species, i, a2, b2,
            slipping_system['a'], slipping_system['b']
        )
        
        axes[i].set_aspect('equal', adjustable='box')
        # optional: tighten the data limits so the box really is square
        axes[i].autoscale(enable=True, axis='both', tight=True)

def plot_all_layers_element_colored(
    struct, 
    ax, 
    miller_a, 
    miller_b,tol=1e-6
    ):
    # 1) 原胞分数坐标和笛卡尔投影

    fracs   = np.array(struct.frac_coords)
    lattice = struct.lattice.matrix
    
    a3, b3  = lattice[1], lattice[2]
    la, lb  = np.linalg.norm(a3), np.linalg.norm(b3)
    costheta    = np.dot(a3, b3)/(la*lb)
    theta       = np.arccos(np.clip(costheta, -1,1))
    B2D     = np.array([[la,    0.0],
                        [lb*costheta,   lb*np.sin(theta)]])

    sab        = fracs[:, [1, 2]]
    cart2d     = sab.dot(B2D)
    vals       = fracs[:, 0]
    rounded    = np.round(vals/tol)*tol
    layers     = np.unique(rounded); layers.sort()
    n_layers   = len(layers)
    alphas     = np.linspace(0.3, 1.0, n_layers)

    species_all    = np.array([site.species_string for site in struct])
    unique_species = list(dict.fromkeys(species_all))
    palette        = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
    color_map      = {sp: palette[i%len(palette)] for i,sp in enumerate(unique_species)}

    radius = min(la, lb)*0.1

    for i, lv in enumerate(layers):
        layer_alpha = alphas[i]
        mask_layer  = np.isclose(rounded, lv, atol=tol)
        pts2d       = cart2d[mask_layer]
        species_l   = species_all[mask_layer]
        for sp in unique_species:
            m_sp = (species_l == sp)
            if not np.any(m_sp): continue
            xs, ys = pts2d[m_sp,0], pts2d[m_sp,1]
            col     = color_map[sp]
            for x, y in zip(xs, ys):
                # 把 layer_alpha 传进去
                draw_sphere(ax, x, y, radius, col, base_alpha=layer_alpha)
                ax.annotate(
                    LIST_ALPHABET[i],
                    xy=(x, y),                      # 原子中心（数据坐标）
                    xytext=(radius+2, radius+2),      # 在半径 + 2pt 处偏移
                    textcoords='offset points',
                    fontsize=16,
                    color='black',
                    weight='bold',
                    ha='center', va='center',
                    zorder=10
                )
    # 绘制平行四边形基矢略…

    # --- 平行四边形基矢 & Miller --- #
    origin = np.zeros(2)
    a2, b2 = B2D[0], B2D[1]
    quad = np.array([origin, a2, a2+b2, b2, origin])
    ax.plot(quad[:,0], quad[:,1], '--', color='black', lw=2, zorder=1)
    # ax.text(a2[0]*1.1, a2[1]*0.9, list_to_tex(miller_a), color='r', va='bottom', ha='left')
    # ax.text(b2[0]*1.1, b2[1]*0.9, list_to_tex(miller_b), color='b', va='bottom', ha='left')
    # ax.text(a2[0]+b2[0], a2[1]+b2[1], list_to_tex(a+b), color='k', va='bottom', ha='left')
    # draw_edge_arrows(ax, a2, b2,
    #              frac_pos=0.5,    # 箭头在 25% 处
    #              arrow_frac=0.2,   # 箭头是边长 20%
    #              offset=-0.3)      # 垂直偏移 0.03 个数据单位
    
    draw_edge_arrow_with_label(
            ax, a2, list_to_tex(miller_a),
            frac_pos=0.5, arrow_frac=0.2, offset=-0.3,
            text_offset=(0.1, 0.1),
            color="black", linewidth=1.5, zorder=15
        )
    draw_edge_arrow_with_label(
            ax, b2, list_to_tex(miller_b),
            frac_pos=0.5, arrow_frac=0.2, offset=-0.3,
            text_offset=(0.1, 0.1),
            color="black", linewidth=1.5, zorder=15
        )
    # --- 美化 --- #
    ax.set_aspect('equal')
    # ax.set_title("All Layers with Sphere Shading")
    # ax.legend(unique_species, loc='upper right', frameon=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    

def deprecated(reason: str):
    def decorator(func):
        message = (
            f"Function {func.__name__!r} is deprecated: {reason}"
        )
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                message,
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@deprecated("use Pymatgen instead")
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

@deprecated("use Pymatgen instead")
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
    

    # 2.
    n = numpy.dot(cut_plane, cell)
    
    n2 = numpy.dot(cell, n)
    # 3.
    fracs = [Fraction(x).limit_denominator(max_denom) for x in n2]
    D = math.lcm(*(f.denominator for f in fracs))
    A, B, C = [int(f * D) for f in fracs]
    g_all = math.gcd(math.gcd(abs(A), abs(B)), abs(C))
    A, B, C = A//g_all, B//g_all, C//g_all

    print(A, B, C)

    g = math.gcd(A, B)
    v1 = numpy.array([ B//g, -A//g, 0 ], dtype=int)
    v2 = numpy.cross(numpy.array([A, B, C], int), v1)

    # 
    def dot(u,v): return int(u.dot(v))
    def norm2(u): return dot(u,u)

    # 5.
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

@deprecated("use Pymatgen instead")
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
        a_mod = alpha - numpy.floor(alpha)
        b_mod = beta  - numpy.floor(beta)
        # 4) 真实空间坐标
        R = x*a + y*b + z*c

        results.append((a_mod, b_mod, layer, R))

    return results

@deprecated("use Pymatgen instead")
def shortest_plane_basis(
    ase_atoms: Atoms,
    cut_plane: list[float],
    tol: float = 1e-6
    ):
    
    v1, v2, v3 = find_plane_pbc(ase_atoms, cut_plane)
    return basis_transform(v1, v2, v3, ase_atoms.positions)

@deprecated("use Pymatgen instead")
def shortest_lattice_and_plane_vectors(
    basis: numpy.ndarray,
    frac_coords: list[float]
) -> tuple[
    numpy.ndarray, tuple[int, int, int],
    list[numpy.ndarray], list[tuple[int, int, int]]
]:
    """
    计算给定晶格基矢和倒易空间坐标法向下：
    1. 法向上的最短晶格向量 d_normal
    2. 平面内的两条最短晶格基矢 d_plane[0], d_plane[1]

    Parameters:
    - basis: shape (3,3) 的 NumPy 数组，每列为晶格基矢 a, b, c。
    - frac_coords: 法向在倒易基底中的坐标 [h, k, l]，可为浮点数或 Fraction。

    Returns:
    - d_normal: 最短晶格向量（笛卡尔坐标）。
    - primitive_hkl_normal: 归一化后的 Miller 整数指数 (h0, k0, l0)。
    - d_plane: List of two np.ndarrays, 平面内最短晶格向量。
    - primitive_hkls_plane: List of two tuples, 平面基矢对应的整数向量 (h, k, l).
    """
    # 将输入坐标转换为 Fraction
    frac_list = []
    for f in frac_coords:
        if isinstance(f, Fraction):
            frac_list.append(f)
        else:
            frac_list.append(Fraction(f).limit_denominator())

    # --- 法向部分 ---
    denoms = [f.denominator for f in frac_list]
    lcm = math.lcm(*denoms)
    hkl_int = [int(f.numerator * (lcm // f.denominator)) for f in frac_list]
    g = math.gcd(math.gcd(abs(hkl_int[0]), abs(hkl_int[1])), abs(hkl_int[2]))
    if g == 0:
        raise ValueError("分数坐标至少需一个非零分量")
    primitive_hkl_normal = tuple(h // g for h in hkl_int)
    d_normal = (primitive_hkl_normal[0] * basis[:, 0] +
                primitive_hkl_normal[1] * basis[:, 1] +
                primitive_hkl_normal[2] * basis[:, 2])

    # --- 平面部分 ---
    M = sympy.Matrix([primitive_hkl_normal])
    nulls = M.nullspace()
    ints = []
    for v in nulls:
        v_rat = sympy.nsimplify(v, [])
        den = [ri.q for ri in v_rat]
        l_val = sympy.ilcm(*den)
        m_int = (v_rat * l_val).applyfunc(lambda x: int(x))
        ints.append(numpy.array(m_int, dtype=int).flatten())
    # 构造候选
    v1, v2 = ints
    candidates = {
        tuple(v1), tuple(v2),
        tuple(v1 + v2), tuple(v1 - v2), tuple(v2 - v1)
    }
    vecs = []
    for hkl in candidates:
        d = hkl[0] * basis[:, 0] + hkl[1] * basis[:, 1] + hkl[2] * basis[:, 2]
        length = numpy.linalg.norm(d)
        vecs.append((length, hkl, d))
    vecs_sorted = sorted(vecs, key=lambda x: x[0])
    _, hkl1, d1 = vecs_sorted[0]
    d_plane = [d1]
    primitive_hkls_plane = [hkl1]
    for length, hkl, d in vecs_sorted[1:]:
        if numpy.linalg.norm(numpy.cross(d1, d)) > 1e-8:
            d_plane.append(d)
            primitive_hkls_plane.append(hkl)
            break

    return d_normal, primitive_hkl_normal, d_plane, primitive_hkls_plane

@deprecated("use Pymatgen instead")
def map_atoms_to_new_cell(old_basis, new_basis, frac_old, padding=1):
    """
    将旧原胞中原子的分数坐标映射到新基矢定义的原胞内，
    并考虑周期性，确保新胞内所有原子都被捕获。

    Parameters:
    - old_basis: (3,3) NumPy 数组，每列是旧基矢
    - new_basis: (3,3) NumPy 数组，每列是新基矢
    - frac_old: (N,3) NumPy 数组，旧原胞内原子的分数坐标
    - padding: 周期包裹范围, 默认1 (即考虑 -1,0,+1 的平移)

    Returns:
    - frac_new_in_cell: (M,3) 新原胞内原子的分数坐标，归一化到 [0,1)
    """
    # 旧分数坐标 -> 笛卡尔
    cart_old = frac_old @ old_basis.T

    # 产生周期平移向量组合
    shifts = numpy.array([[i, j, k] 
                       for i in range(-padding, padding+1)
                       for j in range(-padding, padding+1)
                       for k in range(-padding, padding+1)])

    frac_new_list = []
    inv_new = numpy.linalg.inv(new_basis)

    # 对每个原子和每个平移组合进行映射
    for v_frac, R in zip(frac_old, cart_old):
        for shift in shifts:
            R_shift = R + shift @ old_basis.T
            frac_new = R_shift @ inv_new.T
            # 归一化到[0,1)
            frac_new_mod = frac_new - np.floor(frac_new)
            frac_new_list.append(frac_new_mod)

    frac_new_array = numpy.array(frac_new_list)
    
    # 去重：依靠坐标四舍五入
    unique_frac = numpy.unique(numpy.round(frac_new_array, 6), axis=0)

    return unique_frac

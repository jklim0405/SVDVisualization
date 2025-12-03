import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
from scipy.linalg import svd, expm
import math
from mpl_toolkits.mplot3d import Axes3D

# --------------------------
# 유틸: 로드리게스 회전 및 수학 함수
# --------------------------

def vec_to_skew(a):
    return np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])

def get_rotation_params(R, tol=1e-6):
    """주어진 회전 행렬 R에서 회전축(axis)과 각도(angle)를 계산"""
    trace = np.trace(R)
    cos_theta = (trace - 1) / 2.0
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle = math.acos(cos_theta)
    
    if angle < tol:
        return np.array([1, 0, 0]), 0.0

    R_diff = R - R.T
    sin_theta = np.sin(angle)

    if abs(sin_theta) < tol: 
        if angle < tol: return np.array([1, 0, 0]), 0.0
        vals, vecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(vals - 1.0))
        axis = vecs[:, idx].real
        return axis / np.linalg.norm(axis), angle
    
    axis_raw = np.array([R_diff[2, 1], R_diff[0, 2], R_diff[1, 0]])
    axis = axis_raw / (2 * sin_theta)
    return axis / np.linalg.norm(axis), angle

def rodrigues_interp_matrix(R, t):
    axis, angle = get_rotation_params(R)
    if angle == 0.0: return np.eye(3)
    K = vec_to_skew(axis)
    return expm(t * angle * K)

# --------------------------
# 회전축 계산 안정화
# --------------------------
def rotation_axis_from_matrix(R, tol=1e-6):
    R = np.array(R, dtype=float)
    vals, vecs = np.linalg.eig(R)
    dists = np.abs(vals - 1.0)
    
    idx = int(np.argmin(dists))
    
    axis = vecs[:, idx].real
    norm_val = np.linalg.norm(axis)
    
    if norm_val < 1e-6:
        return np.array([1, 0, 0]), False, 0.0

    axis = axis / norm_val
    
    is_rotation_axis = dists[idx] < 1e-3
    
    return axis, is_rotation_axis, dists[idx]

# --------------------------
# 반사 분해 로직
# --------------------------
def decompose_improper_rotation(M):
    """
    행렬 M (det < 0)을 순수 회전 R과 반사 F로 분해합니다.
    """
    vals, vecs = np.linalg.eig(M)
    
    idx = np.argmin(np.abs(vals + 1.0)) 
    
    normal = vecs[:, idx].real 
    norm_val = np.linalg.norm(normal)
    if norm_val < 1e-12:
        normal = np.array([0, 0, 1]) 
    else:
        normal = normal / norm_val
        
    F = np.eye(3) - 2 * np.outer(normal, normal) 
    R = M @ F 
    
    return R, F, normal

# --------------------------
# 입력 및 SVD 함수 (생략)
# --------------------------
def input_matrix():
    n = 3
    m = 3
    print(f"{n} x {m} 행렬의 각 행을 공백으로 구분하여 입력하세요.")
    rows = []
    for i in range(n):
        while True:
            try:
                row_input = input(f"row {i+1}: ").split()
                if not row_input and i == 0: raise ValueError("Empty")
                if len(row_input) != m:
                    print(f"오류: {m}개의 숫자를 입력해야 합니다.")
                    continue
                rows.append(list(map(float, row_input)))
                break
            except ValueError:
                if i == 0 and not rows: raise
                print("오류: 유효한 숫자를 입력해주세요.")
    return np.array(rows)

def svd_decompose(A):
    U, S, VT = np.linalg.svd(A, full_matrices=True)
    Sigma_full = np.zeros((U.shape[0], VT.shape[0]))
    for i in range(len(S)): Sigma_full[i, i] = S[i]
    return U, Sigma_full, VT, S


# --------------------------
# 애니메이션 주 함수
# --------------------------
def animate_svd_transform(A, v, interval_ms=33, frames_per_transform=75):
    A = np.array(A, dtype=float)
    v = np.array(v, dtype=float).reshape(3)

    U_full, Sigma_full, VT, S = svd_decompose(A)
    U = U_full[:3, :3]
    Sigma3 = np.zeros((3, 3))
    for i in range(len(S)): Sigma3[i, i] = S[i]
    VT3 = VT

    # ---------------------------------------------------------
    # 변환 단계 구성 (회전축 계산 포함)
    # ---------------------------------------------------------
    transforms_processed = []

    # 1. V^T
    if np.linalg.det(VT3) < 0:
        R_pure, F_pure, normal_F = decompose_improper_rotation(VT3)
        axis_R, rel_R, _ = rotation_axis_from_matrix(R_pure)
        transforms_processed.append( ("V^T (Rotation)", R_pure, "$V^T_{rot}$", 1, axis_R, rel_R) ) 
        transforms_processed.append( ("V^T (Reflection)", F_pure, "$V^T_{flip}$", 3, normal_F, None) ) 
    else:
        axis_VT, rel_VT, _ = rotation_axis_from_matrix(VT3)
        transforms_processed.append( ("V^T (Rotation)", VT3, "$V^T$", 1, axis_VT, rel_VT) )

    # 2. Sigma
    transforms_processed.append( ("Sigma (Scaling)", Sigma3, "$V^T \Sigma$", 2, None, None) )

    # 3. U
    if np.linalg.det(U) < 0:
        R_pure, F_pure, normal_F = decompose_improper_rotation(U)
        axis_R, rel_R, _ = rotation_axis_from_matrix(R_pure)
        transforms_processed.append( ("U (Rotation)", R_pure, "$U_{rot}$", 1, axis_R, rel_R) ) 
        transforms_processed.append( ("U (Reflection)", F_pure, "$U_{flip}$", 3, normal_F, None) ) 
    else:
        axis_U, rel_U, _ = rotation_axis_from_matrix(U)
        transforms_processed.append( ("U (Rotation)", U, "$U$", 1, axis_U, rel_U) )


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(bottom=0.20) 

    maxv = max(np.abs(A).max(), np.abs(v).max(), 1.0)
    lim = maxv * 2.0
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("SVD Transform: $A = U \Sigma V^T$ (Reflection Plane & Axis)", fontsize=12, fontweight='bold')

    # 정적 요소
    ax.plot([-lim, lim], [0,0], [0,0], 'k-', lw=0.8, alpha=0.5)
    ax.plot([0,0], [-lim, lim], [0,0], 'k-', lw=0.8, alpha=0.5)
    ax.plot([0,0], [0,0], [-lim, lim], 'k-', lw=0.8, alpha=0.5)

    Av = A @ v
    ax.quiver(0,0,0, v[0], v[1], v[2], color="blue", lw=1.5, ls="-", alpha=0.3, label='Original v')
    ax.quiver(0,0,0, Av[0], Av[1], Av[2], color="green", lw=1.5, ls="-", alpha=0.3, label='Target Av')

    anim_objects = {
        'quiver_main': None,
        'trace_line': None,
        'trace_xs': [], 'trace_ys': [], 'trace_zs': [],
        'quivers_basis': [],
        'info_text': ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, va='top'),
        'step_points': [],
        'step_labels': [],
        'reflection_plane': None, 
        'reflection_edges': [],
        'current_rot_axis_line': None
    }
    anim_objects['trace_line'], = ax.plot([], [], [], color='red', lw=1.2, alpha=0.7)
    
    # --------------------------
    # Checkpoint 계산 
    # --------------------------
    checkpoints = [v]
    curr_cp = v
    for item in transforms_processed:
        curr_cp = item[1] @ curr_cp
        checkpoints.append(curr_cp)
    
    def plot_axis_line(axis_vec, color, lw=2.5, alpha=0.7):
        p = axis_vec * lim * 0.8
        return ax.plot([-p[0], p[0]], [-p[1], p[1]], [-p[2], p[2]], 
                       color=color, lw=lw, ls='--', alpha=alpha)[0]

    
    basis_colors = ['salmon', 'lightgreen', 'lightblue']
    I = np.eye(3)
    total_frames = frames_per_transform * len(transforms_processed)
    
    global current_animation 
    current_animation = None


    def cumulative_before(index):
        C = np.eye(3)
        for j in range(index): C = transforms_processed[j][1] @ C
        return C

    # [수정] 반사 평면 및 테두리를 그리는 함수: 성능 최적화
    def plot_reflection_plane(ax, normal_vec, lim_val, alpha=0.4, color='lightblue', edge_color='darkblue', size_factor=0.7):
        if np.linalg.norm(normal_vec) < 1e-6:
            return None, []

        abs_normal = np.abs(normal_vec)
        main_axis_idx = np.argmax(abs_normal)
        
        # [수정] span의 점 개수를 30개로 줄여 성능 개선 (50 -> 30)
        span = np.linspace(-lim_val * size_factor, lim_val * size_factor, 30) 
        
        epsilon = 1e-6
        if abs(normal_vec[main_axis_idx]) < epsilon:
            return None, []

        if main_axis_idx == 0: 
            y, z = np.meshgrid(span, span)
            x = -(normal_vec[1] * y + normal_vec[2] * z) / normal_vec[0]
            X, Y, Z = x, y, z
        elif main_axis_idx == 1: 
            x, z = np.meshgrid(span, span)
            y = -(normal_vec[0] * x + normal_vec[2] * z) / normal_vec[1]
            X, Y, Z = x, y, z
        else:
            x, y = np.meshgrid(span, span)
            z = -(normal_vec[0] * x + normal_vec[1] * y) / normal_vec[2]
            X, Y, Z = x, y, z
            
        # [수정] rstride, cstride 제거 및 edgecolor='none' 추가: 하나의 부드러운 평면처럼 보이게 함
        # rstride=1, cstride=1로 설정하여 격자를 촘촘하게 그리고, edgecolor='none'으로 격자선을 숨김
        surface = ax.plot_surface(X, Y, Z, alpha=alpha, color=color, rstride=1, cstride=1, 
                                  edgecolor='none', zorder=0)

        edges = []
        # 평면의 경계선(meshgrid의 경계선)을 그리기
        edges.append(ax.plot(X[0, :], Y[0, :], Z[0, :], color=edge_color, lw=1.5, zorder=1)[0]) # 첫 번째 행
        edges.append(ax.plot(X[-1, :], Y[-1, :], Z[-1, :], color=edge_color, lw=1.5, zorder=1)[0]) # 마지막 행
        edges.append(ax.plot(X[:, 0], Y[:, 0], Z[:, 0], color=edge_color, lw=1.5, zorder=1)[0]) # 첫 번째 열
        edges.append(ax.plot(X[:, -1], Y[:, -1], Z[:, -1], color=edge_color, lw=1.5, zorder=1)[0]) # 마지막 열

        return surface, edges


    def update(frame):
        index = frame // frames_per_transform
        
        if index >= len(transforms_processed):
            index = len(transforms_processed) - 1
            t = 1.0
            if hasattr(fig, 'btn_toggle') and fig.anim.event_source is None:
                fig.btn_toggle.label.set_text('Resume')
        else:
            t = (frame % frames_per_transform) / max(1, (frames_per_transform - 1))

        item = transforms_processed[index]
        name, M, label_name, type_id, axis_or_normal_vec, rel_rot = item[0], item[1], item[2], item[3], item[4], item[5]
        
        # ---------------------------------------------------------
        # 위치 및 보간 계산
        # ---------------------------------------------------------
        start_v = checkpoints[index]
        end_v = checkpoints[index+1]
        curr_v = np.zeros(3)

        if type_id == 1: # 회전 (Rodrigues)
            interpM = rodrigues_interp_matrix(M, t)
            curr_v = interpM @ start_v
        elif type_id == 2 or type_id == 3: # 스케일링/반사 (선형 보간)
            curr_v = (1 - t) * start_v + t * end_v

        if t == 1.0: curr_v = end_v

        # ---------------------------------------------------------
        
        # 메인 벡터
        if anim_objects['quiver_main']: anim_objects['quiver_main'].remove()
        anim_objects['quiver_main'] = ax.quiver(0, 0, 0, curr_v[0], curr_v[1], curr_v[2],
                                                color="red", lw=2.5, arrow_length_ratio=0.1)

        # 자취 (생략)
        anim_objects['trace_xs'].append(curr_v[0])
        anim_objects['trace_ys'].append(curr_v[1])
        anim_objects['trace_zs'].append(curr_v[2])
        anim_objects['trace_line'].set_data(anim_objects['trace_xs'], anim_objects['trace_ys'])
        anim_objects['trace_line'].set_3d_properties(anim_objects['trace_zs'])

        # 기저 벡터 (생략)
        current_cumulative_basis = np.eye(3)
        if type_id == 1:
            interpM_basis = rodrigues_interp_matrix(M, t)
            current_cumulative_basis = interpM_basis @ cumulative_before(index)
        else: # Scaling or Reflection
            interpM_basis = np.eye(3) + t * (M - np.eye(3))
            current_cumulative_basis = interpM_basis @ cumulative_before(index)

        for q in anim_objects['quivers_basis']: q.remove()
        anim_objects['quivers_basis'] = []
        for i in range(3):
            bv = current_cumulative_basis @ I[:, i]
            q = ax.quiver(0, 0, 0, bv[0], bv[1], bv[2], color=basis_colors[i], lw=1.0, alpha=0.4, arrow_length_ratio=0.08)
            anim_objects['quivers_basis'].append(q)

        # 텍스트 (생략)
        mat_str = np.array2string(np.round(M, 3), separator=', ', suppress_small=True)
        det_info = f" (det={np.linalg.det(M):.0f})" if type_id==3 else ""
        norm_info_text = f"\nRef Normal: {np.round(axis_or_normal_vec,2)}" if type_id==3 and axis_or_normal_vec is not None else ""
        
        anim_objects['info_text'].set_text(f"Step {index+1}/{len(transforms_processed)}: {name}{det_info}{norm_info_text}\nProgress: {t*100:.1f}%\n\nMatrix:\n{mat_str}")

        # 회전축 표시 (Rotation Axis)
        if anim_objects['current_rot_axis_line']:
            anim_objects['current_rot_axis_line'].remove()
            anim_objects['current_rot_axis_line'] = None
        
        if type_id == 1 and t > 0.1 and t < 0.9: 
            axis_vec = axis_or_normal_vec
            color = 'orange' if 'V^T' in name else 'magenta'
            
            angle = get_rotation_params(M)[1]
            alpha_val = 0.7 if angle > 1e-4 else 0.4 
            
            anim_objects['current_rot_axis_line'] = plot_axis_line(axis_vec, color, alpha=alpha_val)


        # 반사 평면 시각화
        if anim_objects['reflection_plane']: anim_objects['reflection_plane'].remove()
        anim_objects['reflection_plane'] = None 
        for edge in anim_objects['reflection_edges']: edge.remove()
        anim_objects['reflection_edges'] = []
        
        if type_id == 3 and t > 0.1 and t < 0.9: 
            # 성능 최적화가 적용된 함수 호출
            surface, edges = plot_reflection_plane(ax, axis_or_normal_vec, lim, alpha=0.4, color='lightblue', edge_color='darkblue', size_factor=0.7)
            anim_objects['reflection_plane'] = surface
            anim_objects['reflection_edges'].extend(edges)

        # 점 찍기 (생략)
        if t == 1.0 and index >= len(anim_objects['step_points']):
            pt = ax.scatter(curr_v[0], curr_v[1], curr_v[2], color='darkred', s=40, depthshade=False, alpha=0.8)
            anim_objects['step_points'].append(pt)
            lbl_txt = f"{label_name}\n({curr_v[0]:.2f}, {curr_v[1]:.2f}, {curr_v[2]:.2f})"
            lbl = ax.text(curr_v[0], curr_v[1], curr_v[2], lbl_txt, fontsize=8, color='darkred', ha='left')
            anim_objects['step_labels'].append(lbl)

    # --------------------------
    # 실행 및 컨트롤 (생략)
    # --------------------------
    
    def start_animation():
        global current_animation
        
        if current_animation and current_animation.event_source: 
            try: current_animation.event_source.stop()
            except: pass
        
        # 초기화 로직
        for p in anim_objects['step_points']: p.remove()
        for l in anim_objects['step_labels']: l.remove()
        anim_objects['step_points'].clear()
        anim_objects['step_labels'].clear()
        anim_objects['trace_xs'].clear()
        anim_objects['trace_ys'].clear()
        anim_objects['trace_zs'].clear()
        anim_objects['trace_line'].set_data([], [])
        anim_objects['trace_line'].set_3d_properties([])
        
        if anim_objects['reflection_plane']: anim_objects['reflection_plane'].remove()
        anim_objects['reflection_plane'] = None
        
        for edge in anim_objects['reflection_edges']: edge.remove()
        anim_objects['reflection_edges'] = []
        
        if anim_objects['current_rot_axis_line']: anim_objects['current_rot_axis_line'].remove()
        anim_objects['current_rot_axis_line'] = None


        new_anim = FuncAnimation(fig, update, frames=total_frames, interval=interval_ms, blit=False, repeat=False)
        current_animation = new_anim 
        fig.anim = current_animation

        if hasattr(fig, 'btn_toggle'): fig.btn_toggle.label.set_text('Pause')
        
        return current_animation

    ax_toggle = plt.axes([0.35, 0.05, 0.15, 0.075])
    btn_toggle = Button(ax_toggle, 'Pause')
    fig.btn_toggle = btn_toggle

    def toggle_animation(event):
        is_running = fig.anim.event_source and getattr(fig.anim.event_source, '_timer', None)
        if is_running:
            fig.anim.pause()
            btn_toggle.label.set_text('Resume')
        else:
            if fig.anim.event_source is None: start_animation() 
            else:
                fig.anim.resume()
                btn_toggle.label.set_text('Pause')
    btn_toggle.on_clicked(toggle_animation)

    ax_replay = plt.axes([0.8, 0.05, 0.1, 0.075])
    btn_replay = Button(ax_replay, 'Replay')
    btn_replay.on_clicked(lambda e: start_animation())

    initial_anim = start_animation()
    plt.show()

    return initial_anim

if __name__ == "__main__":
    print("=== SVD Visualization (Accurate Reflection Plane) ===")
    try:
        user_input = input("초기 벡터 v (x y z)를 입력하세요 (예: 1 0 0): ")
        v = list(map(float, user_input.split())) if user_input.strip() else [1, 1, 1]
        print("\n변환 행렬 A 입력 (Enter 시 기본 예제 - 반사 포함)")
        try: A = input_matrix()
        except:
             print("기본 예제 사용 (Reflection 포함)")
             # 부정 회전 예제: V^T와 U 모두 부정 회전이 되지 않도록 A 행렬을 설정 (V^T가 det=-1이 될 확률이 높음)
             A = np.array([[-1.5, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]])
             
        global anim_reference 
        anim_reference = animate_svd_transform(A, v, interval_ms=20, frames_per_transform=60)
        
    except Exception as e:
        print(f"\n오류: {e}")
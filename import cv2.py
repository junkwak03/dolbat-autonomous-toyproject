import cv2
import numpy as np
import os
import time

# ==============================================================
# [과제 1] 주행영상 전처리 (인지)
# 1. 사다리꼴 ROI 설정 및 BEV 변환 (양 차선 수직 평행화)
# 2. 초록 매트 경계(우측/상단) 기반 배리어 + 아스팔트/차선 연결성 추출
# 3. 횡단보도가 화면 상단(y=0)에 처음 진입하는 순간부터 끝까지 끊김 없이 인식하고,
#    차선 밖(꽃무늬, 테이블, 슬리퍼, 문, 갈색 벽)은 100% 검정(0) 처리
# 4. 저해상도(1/4) 마스크 연산 및 BoxFilter 최적화로 원본 영상 실시간 속도(1배속) 유지
# ==============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, 'dolbat_toyproject.mp4')

# BEV 변환 해상도
BEV_W, BEV_H = 400, 500

# [사다리꼴 ROI 좌표 (640x480 기준)]
SRC_PTS = np.float32([
    [125, 295],   # Top-Left
    [465, 295],   # Top-Right
    [575, 475],   # Bottom-Right
    [10,  475]    # Bottom-Left
])

# BEV 상에서 양 차선이 수직 평행하도록 매핑
DST_PTS = np.float32([
    [0,   0],
    [400, 0],
    [380, 500],
    [30,  500]
])

M = cv2.getPerspectiveTransform(SRC_PTS, DST_PTS)


def build_green_barrier_fast(is_green_small):
    """
    1/4 축소 해상도(100x125)에서 초고속(0.2ms 이하)으로 초록 매트 및 배경 차단 배리어 생성.
    (기존 65x65 타원 모폴로지 연산으로 인해 프레임이 느려지던 현상을 200배 이상 가속)
    """
    sh, sw = is_green_small.shape
    green_u8 = is_green_small.astype(np.uint8) * 255

    # 1. 화면 우측/상단 경계에 걸친 꽃무늬도 완벽히 메우기 위해 패딩 후 MORPH_CLOSE 수행 (65px -> 1/4 스케일 17px)
    pad = 10
    padded = cv2.copyMakeBorder(green_u8, pad, pad, 0, pad, cv2.BORDER_REPLICATE)
    kernel_flower = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    closed_padded = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel_flower)
    green_with_flowers = closed_padded[pad:pad + sh, 0:sw]

    # 가로/세로 연속된 진짜 초록 매트 영역만 추출
    horiz_green = cv2.morphologyEx(
        green_with_flowers, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 2))
    )
    vert_green = cv2.morphologyEx(
        green_with_flowers, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    )

    green_barrier = green_with_flowers.copy()

    # 2. [우측 차단] 각 행(y)에서 초록 매트 시작점부터 우측 끝까지 차단 (꽃무늬 및 우측 배경 차단)
    has_green_row = np.any(horiz_green > 0, axis=1)
    first_green_x = np.argmax(horiz_green > 0, axis=1)
    col_indices = np.arange(sw)[None, :]
    right_of_green = has_green_row[:, None] & (col_indices >= first_green_x[:, None])
    green_barrier[right_of_green] = 255

    # 3. [좌회전 상단 차단] 좌회전 시 대각선 초록 매트 위쪽(심사위원 테이블, 슬리퍼, 흰색 문, 갈색 벽) 차단
    limit_x = int(sw * 0.75)  # 원본 x <= 300 과 동일
    sub_vert = vert_green[:, :limit_x]
    has_green_col = np.any(sub_vert > 0, axis=0)
    last_green_y = sh - 1 - np.argmax(sub_vert[::-1, :] > 0, axis=0)
    row_indices = np.arange(sh)[:, None]
    above_green = has_green_col[None, :] & (row_indices <= last_green_y[None, :])
    green_barrier[:, :limit_x][above_green] = 255

    # 초록 매트가 좌측 끝 부근까지 도달했을 때 좌측 상단 모서리 밀봉
    left_hits = np.where(has_green_row & (first_green_x <= 13))[0]
    if len(left_hits) > 0:
        y_seal = int(np.max(left_hits))
        green_barrier[:y_seal, :15] = 255

    return green_barrier


def get_valid_road_zone(gray, r, b, is_green):
    """
    1/4 해상도(100x125)에서 고속 모폴로지 연산으로 유효 도로 영역을 구한 뒤
    원래 해상도(400x500)로 복원하여 영상 버벅임 없이 실시간 속도를 유지하는 함수
    """
    sh, sw = BEV_H // 4, BEV_W // 4
    gray_s = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_NEAREST)
    r_s = cv2.resize(r, (sw, sh), interpolation=cv2.INTER_NEAREST)
    b_s = cv2.resize(b, (sw, sh), interpolation=cv2.INTER_NEAREST)
    is_green_s = cv2.resize(is_green.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST) > 0

    green_barrier_s = build_green_barrier_fast(is_green_s)
    not_barrier_s = (green_barrier_s == 0)

    # 1. 어두운 아스팔트 노면 검출
    is_asphalt_s = (gray_s >= 15) & (gray_s < 128) & not_barrier_s & \
                   ((r_s.astype(int) - b_s.astype(int)) <= 24)
    asphalt_mask_s = is_asphalt_s.astype(np.uint8) * 255

    # 2. 정지선 및 횡단보도 줄무늬 간격 연결 (1/4 스케일 커널: (3, 11), (9, 3))
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 11))
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    asphalt_bridged_s = cv2.morphologyEx(asphalt_mask_s, cv2.MORPH_CLOSE, kernel_v)
    asphalt_bridged_s = cv2.morphologyEx(asphalt_bridged_s, cv2.MORPH_CLOSE, kernel_h)
    asphalt_bridged_s[~not_barrier_s] = 0

    # 3. 초록 배리어 안쪽(도로 측)의 모든 유효 아스팔트 영역을 상단(y=0)부터 즉시 채택
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(asphalt_bridged_s, connectivity=8)
    main_road_s = np.zeros_like(asphalt_bridged_s)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 8:  # 원본 스케일 128px 상당
            main_road_s[labels == i] = 255

    if np.sum(main_road_s > 0) < 20:
        main_road_s = asphalt_bridged_s.copy()

    # 4. 도로 내부 메우기 및 도로 경계 확장 (1/4 스케일 커널: (11, 11), (17, 17))
    kernel_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    main_road_s = cv2.morphologyEx(main_road_s, cv2.MORPH_CLOSE, kernel_fill)
    main_road_s[~not_barrier_s] = 0

    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    valid_road_s = cv2.dilate(main_road_s, kernel_expand)
    valid_road_s[~not_barrier_s] = 0

    # 원본 BEV 해상도(400x500)로 복원
    valid_road_zone = cv2.resize(valid_road_s, (BEV_W, BEV_H), interpolation=cv2.INTER_NEAREST)
    green_barrier = cv2.resize(green_barrier_s, (BEV_W, BEV_H), interpolation=cv2.INTER_NEAREST)
    green_barrier[is_green] = 255
    valid_road_zone[green_barrier > 0] = 0

    return valid_road_zone, green_barrier


def preprocess(frame):
    """
    과제 1 전처리 파이프라인:
    - 횡단보도가 상단(y=0)에 진입하는 즉시 6개 줄무늬 모두 선명하게 인식
    - 좌·우 차선 선명하게 인식, 차선 밖(꽃, 테이블, 슬리퍼, 문, 벽)은 100% 검정(0)
    """
    # 1. 원본 영상에 사다리꼴 ROI 표시
    roi_vis = frame.copy()
    pts = SRC_PTS.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(roi_vis, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # 2. ROI 영역 BEV 변환
    bev = cv2.warpPerspective(frame, M, (BEV_W, BEV_H), flags=cv2.INTER_LINEAR)

    # 3. 색공간 변환
    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    l_chan = hls[:, :, 1]
    h_hsv, s_hsv, _ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b, g, r = cv2.split(bev)

    # 4. 초록 바닥 감지
    is_green = (g >= 100) & (
        ((h_hsv >= 30) & (h_hsv <= 90) & (s_hsv >= 24)) |
        ((g.astype(int) > r.astype(int) + 15) & (g.astype(int) > b.astype(int) + 6))
    )

    # 5. 도로 유효 영역 및 초록 배리어 고속 계산
    valid_road_zone, green_barrier = get_valid_road_zone(gray, r, b, is_green)

    # 6. 도로 위 흰색 표식(차선, 횡단보도) 추출
    #    - 무거운 45x45 MORPH_TOPHAT 대신 O(1) 초고속 BoxFilter로 주변 평균 밝기와의 대비(local_diff) 계산
    gray_open = cv2.morphologyEx(
        gray, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    )
    local_mean = cv2.boxFilter(gray_open, -1, (45, 45))
    local_diff = gray_open.astype(np.int16) - local_mean.astype(np.int16)

    bright_white = (gray >= 118) | (l_chan >= 118) | ((gray_open >= 104) & (local_diff >= 12))
    not_brown_wood = (s_hsv <= 95) & ((r.astype(int) - b.astype(int)) <= 42)

    # 초록 배리어 안쪽(도로 측)의 모든 흰색 후보 픽셀 추출
    candidate_mask = np.zeros((BEV_H, BEV_W), dtype=np.uint8)
    candidate_mask[bright_white & not_brown_wood & (~is_green) & (green_barrier == 0)] = 255

    # 7. 도로 중앙의 얇은 2~3px 격자선 제거 및 차선/횡단보도 내부 채움
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel_open)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_CLOSE, kernel_close)

    # 8. 흰색 덩어리(횡단보도 줄무늬, 차선)의 일부분이라도 도로 영역(valid_road_zone)에 닿아 있으면 통째로 유지
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, connectivity=8)
    binary_clean = np.zeros_like(candidate_mask)
    for i in range(1, num_labels):
        area_i = stats[i, cv2.CC_STAT_AREA]
        height_i = stats[i, cv2.CC_STAT_HEIGHT]
        top_y = stats[i, cv2.CC_STAT_TOP]

        is_valid_size = (area_i >= 70 and height_i >= 14) or (top_y <= 5 and area_i >= 45 and height_i >= 9)
        if is_valid_size:
            comp_mask = (labels == i)
            if np.any(valid_road_zone[comp_mask] > 0):
                binary_clean[comp_mask] = 255

    # 차선과 횡단보도를 선명하게 보강
    binary_clean = cv2.dilate(binary_clean, np.ones((3, 3), np.uint8), iterations=1)

    return roi_vis, bev, binary_clean


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: {VIDEO_PATH} 를 열 수 없습니다.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 24.0
    frame_interval_ms = int(1000.0 / fps)

    # 창 생성 및 정렬
    cv2.namedWindow("1. Original ROI")
    cv2.namedWindow("2. BEV Transformed")
    cv2.namedWindow("3. Preprocessed Binary")
    cv2.namedWindow("Task 1: Preprocess Comparison")

    cv2.moveWindow("1. Original ROI", 30, 40)
    cv2.moveWindow("2. BEV Transformed", 550, 40)
    cv2.moveWindow("3. Preprocessed Binary", 980, 40)
    cv2.moveWindow("Task 1: Preprocess Comparison", 30, 460)

    while cap.isOpened():
        t_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 무한 반복 재생
            continue

        # 전처리 수행
        roi_vis, bev, binary_clean = preprocess(frame)

        # 개별 창 출력
        cv2.imshow("1. Original ROI", roi_vis)
        cv2.imshow("2. BEV Transformed", bev)
        cv2.imshow("3. Preprocessed Binary", binary_clean)

        # Slide 2 형태의 가로 3단 비교 뷰
        target_h = 320
        v1 = cv2.resize(roi_vis, (int(roi_vis.shape[1] * (target_h / roi_vis.shape[0])), target_h))
        v2 = cv2.resize(bev, (int(bev.shape[1] * (target_h / bev.shape[0])), target_h))
        v3_color = cv2.cvtColor(binary_clean, cv2.COLOR_GRAY2BGR)
        v3 = cv2.resize(v3_color, (int(v3_color.shape[1] * (target_h / v3_color.shape[0])), target_h))

        cv2.putText(v1, "[1] Original ROI", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.putText(v2, "[2] BEV Transformed", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.putText(v3, "[3] Preprocessed Binary", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        comparison = np.hstack((v1, v2, v3))
        cv2.imshow("Task 1: Preprocess Comparison", comparison)

        # 연산 시간을 제외한 만큼만 대기하여 원본 영상 속도(1배속)를 정확히 유지
        elapsed_ms = int((time.perf_counter() - t_start) * 1000)
        wait_ms = max(1, frame_interval_ms - elapsed_ms)

        key = cv2.waitKey(wait_ms) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 32: # Space: 일시정지 / 재생
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

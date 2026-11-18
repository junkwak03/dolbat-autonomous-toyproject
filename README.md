[main.py](https://github.com/user-attachments/files/35037472/main.py)
"""양쪽 차선을 보존하는 ROI 전처리. 실행: python main.py --headless"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from lane_detection import SlidingWindowTracker, lane_candidates

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / 'dolbat_toyproject.mp4'
OUT_COMPARISON_PATH = BASE_DIR / 'result_comparison.mp4'
OUT_BINARY_PATH = BASE_DIR / 'result_binary.mp4'
BEV_W, BEV_H = 640, 500
# 화면 중심 x=319.5를 기준으로 좌우 대칭인 사다리꼴.
# 양쪽 빗변의 가로 여백을 80px로 맞춰 한쪽으로 기울지 않게 한다.
# 640x480 기준이며 입력 해상도에 맞춰 비례 조정한다.
SRC_PTS = np.float32([[80, 260], [559, 260], [639, 479], [0, 479]])
# 과제 예시처럼 ROI의 네 꼭짓점을 BEV 직사각형의 네 모서리에 대응한다.
DST_PTS = np.float32([[0, 0], [BEV_W - 1, 0],
                      [BEV_W - 1, BEV_H - 1], [0, BEV_H - 1]])


def perspective_matrix(shape):
    h, w = shape[:2]
    source = SRC_PTS * np.float32([w / 640, h / 480])
    return cv2.getPerspectiveTransform(source, DST_PTS), source


def save_image_safe(filename, img):
    path = BASE_DIR / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix, img)
    if not ok:
        raise OSError(f'Image encoding failed: {path}')
    encoded.tofile(str(path))
    return path


def road_support(bev):
    """가장 큰 어두운 무채색 노면 성분만 선택한다. 특정 매트 색은 사용하지 않는다."""
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    dark = ((gray > 15) & (gray < 135) & (hsv[:, :, 1] < 100)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if count <= 1:
        return np.zeros_like(dark)
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    road = (labels == largest).astype(np.uint8) * 255
    # 노면 내부의 흰 도색 구멍은 채우되 바깥 물체 쪽으로 크게 확장하지 않는다.
    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(road, contours, -1, 255, cv2.FILLED)
    return cv2.dilate(road, np.ones((9, 9), np.uint8))


def white_candidates(bev):
    """매트의 색이나 위치 대신 밝기·채도와 주변 대비로 흰색 도색을 추출한다."""
    hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
    local = cv2.GaussianBlur(gray, (31, 31), 0)
    contrast = gray.astype(np.int16) - local.astype(np.int16)
    # 밝은 유색 바닥은 회색 밝기가 높아도 흰색 도색이 아니다.
    # 흰 차선 중심의 채도는 대부분 10 미만이다. 밝은 매트 경계도 제외한다.
    white = (hsv[:, :, 1] < 20) & (gray > 125) & ((contrast > 10) | (gray > 190))
    support_mask = road_support(bev)
    # 노면 안쪽에서는 압축·그림자로 흐려진 가는 표식도 보존한다.
    # 노면 경계와 바깥은 엄격한 흰색 조건을 유지해 매트 재검출을 억제한다.
    interior = cv2.erode(support_mask, np.ones((21, 21), np.uint8)) > 0
    white |= interior & (hsv[:, :, 1] < 70) & (gray > 110) & ((contrast > 8) | (gray > 190))
    # 특정 색 경계 대신 도색 주변의 어두운 무채색 노면을 확인한다.
    # 가늘거나 짧거나 비스듬한 노면 표시도 포함한다. Opening과 방향 필터를
    # 적용하면 주차 구획선·유도선·문양의 일부가 지워지므로 성분 잡음만 제거한다.
    candidate = white.astype(np.uint8) * 255
    support = support_mask > 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    accepted = np.zeros_like(candidate)
    for label in range(1, count):
        component = labels == label
        if stats[label, cv2.CC_STAT_AREA] >= 12 and np.count_nonzero(support[component]) >= .08 * stats[label, cv2.CC_STAT_AREA]:
            accepted[component] = 255
    # 유효한 흰 도색 성분은 내부까지 보존한다. 노면과의 거리로 픽셀을
    # 다시 자르면 두꺼운 차선 중앙이 지워져 테두리 두 줄만 남을 수 있다.
    return accepted


def preprocess(frame, tracker=None):
    matrix, source = perspective_matrix(frame.shape)
    roi_vis = frame.copy()
    roi_mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.fillConvexPoly(roi_mask, source.astype(np.int32), 255)
    roi_vis[roi_mask == 0] = (roi_vis[roi_mask == 0] * .3).astype(np.uint8)
    cv2.polylines(roi_vis, [source.astype(np.int32)], True, (0, 0, 255), 2)
    masked = cv2.bitwise_and(frame, frame, mask=roi_mask)
    # 모든 출력 픽셀이 ROI 내부에 대응하므로 ROI 바깥 검정색이 섞이지 않게 한다.
    bev = cv2.warpPerspective(frame, matrix, (BEV_W, BEV_H))
    # 흰 도색 후보를 BEV로 변환한 뒤 횡단보도·정지선·가는 구획선을 제외한다.
    original_white = white_candidates(masked)
    # BEV에서 확대된 주차선이 차선 두께로 보이지 않도록 원본에서 먼저 거른다.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(original_white, 8)
    thickness = cv2.distanceTransform(original_white, cv2.DIST_L2, 5)
    for label in range(1, count):
        component = labels == label
        if np.percentile(thickness[component], 80) < 2.2 * frame.shape[1] / 640:
            original_white[component] = 0
    warped_white = cv2.warpPerspective(original_white, matrix, (BEV_W, BEV_H),
                                        flags=cv2.INTER_NEAREST)
    candidates = lane_candidates(warped_white)
    detector = tracker if tracker is not None else SlidingWindowTracker()
    detector.process(candidates, prefiltered=True)
    binary = detector.last_lane_mask
    return roi_vis, bev, binary


def comparison_image(roi, bev, binary):
    panels = []
    for img, title in ((roi, '1. Trapezoid ROI'), (bev, '2. Perspective view'),
                       (cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), '3. Lane binary')):
        panel = cv2.resize(img, (426, 320))
        cv2.rectangle(panel, (0, 0), (425, 31), (0, 0, 0), -1)
        cv2.putText(panel, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        panels.append(panel)
    return np.hstack(panels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='창 없이 전체 영상을 한 번 처리')
    parser.add_argument('--input', type=Path, default=VIDEO_PATH)
    parser.add_argument('--output-dir', type=Path, default=BASE_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise OSError(f'Cannot open video: {args.input}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 24.0
    writers = []
    tracker = SlidingWindowTracker()
    detection_log = None
    frames = 0
    started = time.perf_counter()
    try:
        for name, size, color in [('result_comparison.mp4', (1704, 320), True),
                                  ('result_binary.mp4', (BEV_W, BEV_H), False),
                                  ('result_sliding_window.mp4', (BEV_W, BEV_H), True)]:
            writer = cv2.VideoWriter(str(args.output_dir / name), cv2.VideoWriter_fourcc(*'mp4v'), fps, size, color)
            writers.append(writer)
            if not writer.isOpened():
                raise OSError(f'Cannot create video: {name}')
        detection_log = (args.output_dir / 'result_lanes.jsonl').open('w', encoding='utf-8')
        while True:
            frame_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                break
            roi, bev, binary = preprocess(frame, tracker)
            sliding, lanes = tracker.last_visual, tracker.last_lanes
            panel = cv2.resize(sliding, (426, 320))
            cv2.rectangle(panel, (0, 0), (425, 31), (0, 0, 0), -1)
            status = ' '.join(f'{side[0].upper()}:{"OK" if lane is not None else "LOST"}'
                              for side, lane in lanes.items())
            cv2.putText(panel, f'4. Sliding windows {status}', (6, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
            comparison = np.hstack((comparison_image(roi, bev, binary), panel))
            writers[0].write(comparison)
            writers[1].write(binary)
            writers[2].write(sliding)
            detection_log.write(json.dumps(dict(frame=frames, time_s=frames / fps,
                left=None if lanes['left'] is None else lanes['left'].as_dict(),
                right=None if lanes['right'] is None else lanes['right'].as_dict())) + '\n')
            frames += 1
            if not args.headless:
                cv2.imshow('Lane preprocessing', comparison)
                key = cv2.waitKey(max(1, int(1000 / fps - 1000 * (time.perf_counter() - frame_start)))) & 255
                if key == 32:
                    key = cv2.waitKey(0) & 255
                if key == ord('s'):
                    save_image_safe(f'capture_{frames:04d}_comparison.png', comparison)
                    save_image_safe(f'capture_{frames:04d}_binary.png', binary)
                if key in (27, ord('q')):
                    break
    finally:
        cap.release()
        for writer in writers:
            writer.release()
        if detection_log is not None:
            detection_log.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f'Saved {frames} frames ({fps:g} fps), processing time {time.perf_counter() - started:.1f}s')
    print(args.output_dir / 'result_comparison.mp4')
    print(args.output_dir / 'result_binary.mp4')
    print(args.output_dir / 'result_sliding_window.mp4')


if __name__ == '__main__':
    main()


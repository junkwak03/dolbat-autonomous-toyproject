"""과제 2: 전처리 결과에서 좌우 차선을 Sliding Window로 각각 추적한다."""
from dataclasses import dataclass

import cv2
import numpy as np


def lane_candidates(binary):
    """횡단보도·정지선·가는 구획선을 제외한 차선 전처리 마스크를 만든다."""
    clean = binary.copy()
    repeated = np.zeros_like(clean)
    h, w = clean.shape
    # 정지선이 실선에 붙어 있어도 가로 선분만 끊어 짧은 가지가 남지 않게 한다.
    lines = cv2.HoughLinesP(clean, 1, np.pi / 180, 35,
                            minLineLength=max(60, int(w * .2)), maxLineGap=20)
    if lines is not None:
        transverse = np.zeros_like(clean)
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            dx, dy = int(x2 - x1), int(y2 - y1)
            if dx and abs(dy) <= .2 * abs(dx):
                extension = 25 if dx > 0 else -25
                slope = dy / dx
                start = (int(x1 - extension), int(y1 - slope * extension))
                end = (int(x2 + extension), int(y2 + slope * extension))
                cv2.line(transverse, start, end, 255, max(7, h // 70))
        clean[transverse > 0] = 0
    for y, row in enumerate(clean):
        edges = np.diff(np.r_[0, (row > 0).astype(np.int8), 0])
        starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
        stripes = [(a, b) for a, b in zip(starts, ends) if 12 <= b - a <= 90]
        for i in range(len(stripes) - 2):
            group = stripes[i:i + 3]
            widths = np.array([b - a for a, b in group])
            centers = np.array([(a + b) / 2 for a, b in group])
            gaps = np.diff(centers)
            if widths.max() < 1.6 * widths.min() and gaps.max() < 1.5 * gaps.min() and gaps.min() > widths.mean() * 1.3:
                for a, b in group:
                    repeated[y, a:b] = 255
                # 화면 가장자리에 일부만 보이는 줄무늬도 반복 간격으로 확인한다.
                period = float(np.median(gaps))
                stripe_width = float(np.median(widths))
                for a, b in zip(starts, ends):
                    center = (a + b) / 2
                    phase = round((center - centers[0]) / period)
                    predicted = centers[0] + phase * period
                    overlap = max(0, min(b, predicted + stripe_width / 2) -
                                  max(a, predicted - stripe_width / 2))
                    if overlap >= .7 * (b - a):
                        repeated[y, a:b] = 255
        for a, b in zip(starts, ends):
            if b - a < 3 or b - a > w * .12:
                row[a:b] = 0
    # 가는 주차선이 교차점 때문에 큰 성분으로 연결돼 있어도 연결 목을 끊는다.
    # 작은 가로 정지선 잔여물도 제거하며 두꺼운 차선 내부는 유지한다.
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    distance = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    for label in range(1, count):
        component = labels == label
        _, _, width, height, area = stats[label]
        short_transverse = height < 36 and width > 1.6 * height
        if (area < 40 or height < 18 or short_transverse or np.max(distance[component]) < 2.8 or
                np.count_nonzero(repeated[component]) > .3 * area):
            clean[component] = 0
    return clean


@dataclass
class Lane:
    coefficients: np.ndarray
    points: np.ndarray
    windows: list
    y_min: int
    y_max: int
    residual: float
    curvature_source: str = 'observed'

    def x_at(self, y):
        return np.polyval(self.coefficients, y)

    def as_dict(self):
        return dict(coefficients=self.coefficients.tolist(), observed_y=[self.y_min, self.y_max],
                    point_rows=len(self.points), residual_px=round(self.residual, 2),
                    curvature_source=self.curvature_source)


def fit_lane_curve(points, previous, height):
    """일관된 2차식으로 맞추되 짧은 관측의 곡률은 이전 값에 가깝게 제한한다.

    관측 범위가 넓어질수록 제약을 연속적으로 줄여 120px에서 차수가
    갑자기 바뀌지 않는다. 위치와 접선 기울기는 현재 관측으로 갱신한다.
    """
    y, x = points[:, 0], points[:, 1]
    center = float(np.mean(y))
    scale = float(height)
    t = (y - center) / scale
    design = np.c_[t * t, t, np.ones_like(t)]
    prior_a = 0.0 if previous is None else float(np.pad(previous.coefficients,
        (max(0, 3 - len(previous.coefficients)), 0))[0])
    span = max(35.0, float(np.ptp(y)))
    strength = len(y) * .001 * (120 / span) ** 4
    weight = np.sqrt(strength)
    design = np.vstack((design, [weight, 0, 0]))
    target = np.r_[x, weight * prior_a * scale * scale]
    a, b, c = np.linalg.lstsq(design, target, rcond=None)[0]
    return np.array([a / scale ** 2,
                     b / scale - 2 * a * center / scale ** 2,
                     c - b * center / scale + a * center ** 2 / scale ** 2])


def selected_lane_pixels(binary, lanes):
    """현재 프레임에서 추적한 차선과 일치하는 실제 흰 픽셀만 남긴다.

    피팅 곡선을 흰색으로 그려 넣지 않으며, 점선 공백이나 미검출 차선을
    채우지 않는다. 곡선 근처라도 별도 흰 띠와 가로 가지는 제외한다.
    """
    result = np.zeros_like(binary)
    for lane in lanes.values():
        if lane is None:
            continue
        matches = []
        tolerance = min(12, max(6, lane.residual * 3))
        for y, observed_x in lane.points:
            y = int(y)
            prediction = float(lane.x_at(y))
            if abs(observed_x - prediction) > tolerance:
                continue
            edges = np.diff(np.r_[0, (binary[y] > 0).astype(np.int8), 0])
            starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
            if not len(starts):
                continue
            centers = (starts + ends - 1) / 2
            nearest = int(np.argmin(np.abs(centers - prediction)))
            if abs(centers[nearest] - prediction) <= tolerance:
                matches.append((y, int(starts[nearest]), int(ends[nearest])))
        if not matches:
            continue
        typical_width = float(np.percentile([b - a for _, a, b in matches], 60))
        for y, a, b in matches:
            if b - a <= max(8, typical_width * 1.7):
                result[y, a:b] = binary[y, a:b]
    return result


class SlidingWindowTracker:
    def __init__(self, windows=12, margin=48):
        self.window_count = windows
        self.margin = margin
        self.previous = {'left': None, 'right': None}
        self.missing = {'left': 0, 'right': 0}
        self.last_lane_mask = None
        self.last_visual = None
        self.last_lanes = {'left': None, 'right': None}

    def _seed(self, mask, side):
        h, w = mask.shape
        histogram = ((mask > 0) * np.linspace(.5, 1, h)[:, None]).sum(axis=0)
        histogram = np.convolve(histogram, np.ones(9) / 9, mode='same')
        previous = self.previous[side]
        if previous is not None:
            # 하단에 관측이 없는 점선도 마지막 관측 높이에서 탐색을 재개한다.
            reference_y = min(h - 1, previous.y_max)
            predicted = int(np.clip(previous.x_at(reference_y), 0, w - 1))
            low, high = max(0, predicted - 90), min(w, predicted + 91)
            if high > low and histogram[low:high].max() >= 6:
                return low + int(np.argmax(histogram[low:high]))
        low, high = (0, w // 2) if side == 'left' else (w // 2, w)
        if histogram[low:high].max() < 6:
            return None
        return low + int(np.argmax(histogram[low:high]))

    def _search(self, mask, side):
        h, w = mask.shape
        current = self._seed(mask, side)
        if current is None:
            return None
        points, windows, observed = [], [], []
        bounds = np.linspace(h, 0, self.window_count + 1).astype(int)
        for i in range(self.window_count):
            high, low = bounds[i], bounds[i + 1]
            if len(observed) >= 2:
                y1, x1 = observed[-2]
                y2, x2 = observed[-1]
                slope = np.clip((x2 - x1) / (y2 - y1), -1.5, 1.5)
                current = int(np.clip(x2 + slope * ((low + high) / 2 - y2), 0, w - 1))
            elif observed and self.previous[side] is not None:
                previous = self.previous[side]
                last_y, last_x = observed[-1]
                offset = last_x - previous.x_at(last_y)
                current = int(np.clip(previous.x_at((low + high) / 2) + offset, 0, w - 1))
            x_low, x_high = max(0, current - self.margin), min(w, current + self.margin + 1)
            band = mask[low:high, x_low:x_high]
            histogram = (band > 0).sum(axis=0).astype(float)
            if len(histogram) == 0:
                continue
            histogram = np.convolve(histogram, np.ones(5) / 5, mode='same')
            distance = np.abs(np.arange(len(histogram)) + x_low - current)
            score = histogram / (1 + distance / self.margin)
            peak = x_low + int(np.argmax(score))
            yy, xx = np.nonzero(band)
            xx = xx + x_low
            yy = yy + low
            selected = np.abs(xx - peak) <= 22
            xx, yy = xx[selected], yy[selected]
            found = len(xx) >= 25 and len(np.unique(yy)) >= max(8, int((high - low) * .25))
            windows.append((x_low, low, x_high - 1, high - 1, bool(found)))
            if found:
                row_points = [(int(y), float(np.median(xx[yy == y]))) for y in np.unique(yy)]
                points.extend(row_points)
                current = int(np.median(xx))
                observed.append((float(np.median(yy)), float(current)))
        if len(points) < 20:
            return None
        points = np.asarray(points)
        # 큰 점선과 멀리 떨어진 5~10행짜리 조각이 곡선 끝점을 잡지 못하게 한다.
        points = points[np.argsort(points[:, 0])]
        groups = np.split(points, np.flatnonzero(np.diff(points[:, 0]) > 3) + 1)
        minimum_rows = max(12, min(24, .15 * max(len(group) for group in groups)))
        groups = [group for group in groups if len(group) >= minimum_rows]
        if not groups:
            return None
        points = np.vstack(groups)
        if len(points) < 20:
            return None
        if np.ptp(points[:, 0]) < 35:
            return None
        previous = self.previous[side]
        for _ in range(3):
            fit = fit_lane_curve(points, previous, h)
            residuals = np.abs(points[:, 1] - np.polyval(fit, points[:, 0]))
            inliers = residuals < max(8, 2.5 * float(np.median(residuals)))
            if inliers.all() or inliers.sum() < 20:
                break
            points = points[inliers]
        if len(points) < 20 or np.ptp(points[:, 0]) < 35:
            return None
        fit = fit_lane_curve(points, previous, h)
        residual = float(np.median(np.abs(points[:, 1] - np.polyval(fit, points[:, 0]))))
        if residual > 10:
            return None
        windows = [(x1, y1, x2, y2, bool(np.count_nonzero(
            (points[:, 0] >= y1) & (points[:, 0] <= y2)) >= max(8, int((y2-y1+1) * .25))))
            for x1, y1, x2, y2, _ in windows]
        source = 'previous' if previous is not None and np.ptp(points[:, 0]) < 180 else 'observed'
        if previous is None and np.ptp(points[:, 0]) < 180:
            source = 'weak_observation'
        return Lane(fit, points, windows, int(points[:, 0].min()), int(points[:, 0].max()), residual, source)

    def process(self, binary, *, prefiltered=False):
        # 과제 1의 최종 마스크를 받으면 다시 필터링해 차선을 깎지 않는다.
        mask = binary.copy() if prefiltered else lane_candidates(binary)
        lanes = {side: self._search(mask, side) for side in ('left', 'right')}
        left, right = lanes['left'], lanes['right']
        if left is not None and right is not None:
            low, high = max(left.y_min, right.y_min), min(left.y_max, right.y_max)
            if high >= low:
                ys = np.linspace(low, high, 20)
                if np.min(right.x_at(ys) - left.x_at(ys)) < binary.shape[1] * .12:
                    # 같은 선을 좌우로 중복 표시하지 않고 관측이 많은 쪽만 남긴다.
                    lanes['left' if len(left.points) < len(right.points) else 'right'] = None
        for side, lane in lanes.items():
            self.missing[side] = self.missing[side] + 1 if lane is None else 0
            if lane is not None:
                self.previous[side] = lane
            elif self.missing[side] >= 4:
                self.previous[side] = None
        self.last_lane_mask = selected_lane_pixels(mask, lanes)
        visual = cv2.cvtColor(self.last_lane_mask, cv2.COLOR_GRAY2BGR)
        for side, lane in lanes.items():
            color = (255, 100, 0) if side == 'left' else (0, 0, 255)
            if lane is None:
                continue
            for x1, y1, x2, y2, found in lane.windows:
                if y2 < lane.y_min or y1 > lane.y_max:
                    continue
                if not found:
                    y1, y2 = max(y1, lane.y_min), min(y2, lane.y_max)
                cv2.rectangle(visual, (x1, y1), (x2, y2), color if found else (60, 60, 60), 1)
            for y, x in lane.points[::8]:
                cv2.circle(visual, (int(x), int(y)), 3, color, -1)
            ys = np.arange(lane.y_min, lane.y_max + 1)
            xs = lane.x_at(ys)
            valid = (xs >= 0) & (xs < binary.shape[1])
            poly = np.c_[xs[valid], ys[valid]].astype(np.int32)
            cv2.polylines(visual, [poly], False, color, 2)
        self.last_visual = visual
        self.last_lanes = lanes
        return visual, lanes

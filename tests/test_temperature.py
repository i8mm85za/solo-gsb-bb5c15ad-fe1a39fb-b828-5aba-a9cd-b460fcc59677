"""temperature 单元测试：固定用例 + 独立暴力实现的随机交叉验证。"""

import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.temperature import longest_qualified_period  # noqa: E402


def _brute_longest(samples, low, high, max_interval):
    """独立实现：收集全部临界时刻（采样点与温区边界穿越点），对相邻临界点
    构成的基本区间取中点判定是否合格，合并相邻合格基本区间后取最长。"""
    points = set()
    segs = []
    for i in range(len(samples) - 1):
        t0, temp0 = samples[i]
        t1, temp1 = samples[i + 1]
        if t1 - t0 > max_interval:
            continue  # 间隔超限：该段不插值
        segs.append((t0, temp0, t1, temp1))
        points.add(Fraction(t0))
        points.add(Fraction(t1))
        d_temp = temp1 - temp0
        if d_temp:
            for bound in (low, high):
                s = Fraction(bound - temp0, d_temp)
                if 0 < s < 1:
                    points.add(Fraction(t0) + s * (t1 - t0))
    pts = sorted(points)

    def temp_at(x):
        for t0, temp0, t1, temp1 in segs:
            if t0 <= x <= t1:
                return Fraction(temp0) + Fraction(temp1 - temp0) * (x - t0) / (t1 - t0)
        return None

    best_start = best_end = None
    best_dur = Fraction(0)
    cur = None  # 当前未结算的合格连续段 (起, 止)

    def flush():
        nonlocal best_start, best_end, best_dur, cur
        if cur is None:
            return
        dur = cur[1] - cur[0]
        if dur > 0 and (
            dur > best_dur or (dur == best_dur and cur[0] < best_start)
        ):
            best_start, best_end, best_dur = cur[0], cur[1], dur
        cur = None

    for a, b in zip(pts, pts[1:]):
        temp = temp_at((a + b) / 2)
        if temp is not None and low <= temp <= high:
            cur = (a, b) if cur is None else (cur[0], b)
        else:
            flush()
    flush()

    if best_dur <= 0:
        return None, None, Fraction(0)
    return best_start, best_end, best_dur


def test_linear_crossing_exact_fraction():
    # 温度 100→200→300，温区 [150, 250]：t=5 进入，t=15 离开
    got = longest_qualified_period(
        [(0, 100), (10, 200), (20, 300)], 150, 250, 10
    )
    assert got == (Fraction(5), Fraction(15), Fraction(10))


def test_fraction_reduced():
    # 温度 0→10 历时 3 秒，温区 [1, 9]：起 3/10、止 27/10、时长 24/10 = 12/5
    start, end, duration = longest_qualified_period([(0, 0), (3, 10)], 1, 9, 5)
    assert start == Fraction(3, 10)
    assert end == Fraction(27, 10)
    assert duration == Fraction(12, 5)
    assert (duration.numerator, duration.denominator) == (12, 5)


def test_interval_exceeding_max_breaks_curve():
    # (10,200)→(25,300) 间隔 15 > 10：断开不插值，仅首段贡献 [5, 10]
    got = longest_qualified_period(
        [(0, 100), (10, 200), (25, 300)], 150, 250, 10
    )
    assert got == (Fraction(5), Fraction(10), Fraction(5))


def test_interval_equal_to_max_stays_connected():
    # 间隔恰等于上限不断开：与 test_linear_crossing_exact_fraction 同结果
    got = longest_qualified_period(
        [(0, 100), (10, 200), (20, 300)], 150, 250, 10
    )
    assert got == (Fraction(5), Fraction(15), Fraction(10))


def test_constant_segment_in_zone():
    got = longest_qualified_period([(0, 150), (5, 150)], 150, 250, 10)
    assert got == (Fraction(0), Fraction(5), Fraction(5))


def test_boundary_touch_point_has_no_positive_duration():
    # 温度仅在 t=10 触及温区边界 150，时长为 0 → 起止为空
    got = longest_qualified_period(
        [(0, 100), (10, 150), (20, 100)], 150, 250, 10
    )
    assert got == (None, None, Fraction(0))


def test_single_sample_no_extrapolation():
    got = longest_qualified_period([(5, 150)], 100, 200, 10)
    assert got == (None, None, Fraction(0))


def test_all_out_of_zone():
    got = longest_qualified_period(
        [(0, 300), (10, 400), (20, 350)], 100, 200, 10
    )
    assert got == (None, None, Fraction(0))


def test_empty_zone():
    got = longest_qualified_period([(0, 150), (10, 150)], 200, 100, 10)
    assert got == (None, None, Fraction(0))


def test_tie_takes_earliest_start():
    # 两段等长合格时段 [0,5] 与 [15,20]，取开始最早者
    got = longest_qualified_period(
        [(0, 150), (5, 150), (10, 100), (15, 150), (20, 150)],
        150,
        250,
        10,
    )
    assert got == (Fraction(0), Fraction(5), Fraction(5))


def test_gap_inside_otherwise_qualified_curve():
    # 中段温度跌出温区，合格时段被截为两段：
    # [0, 40/3]（t=40/3 时降至下界 150）与 [80/3, 35]，取较长者
    got = longest_qualified_period(
        [(0, 200), (10, 200), (20, 50), (30, 200), (35, 200)],
        150,
        250,
        15,
    )
    assert got == (Fraction(0), Fraction(40, 3), Fraction(40, 3))


def test_random_cross_check():
    rng = random.Random(20260925)
    for case in range(500):
        n = rng.randint(1, 8)
        times = sorted(rng.sample(range(0, 40), n))
        samples = [(t, rng.randint(-50, 350)) for t in times]
        low = rng.randint(-50, 300)
        high = low + rng.randint(0, 100)
        if rng.random() < 0.05:  # 偶发空温区（下界 > 上界）
            low, high = high + 1, low
        max_interval = rng.randint(1, 12)
        got = longest_qualified_period(samples, low, high, max_interval)
        want = _brute_longest(samples, low, high, max_interval)
        assert got == want, (case, samples, low, high, max_interval, got, want)


if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS {name}")
    print(f"\n全部 {len(tests)} 个测试通过")

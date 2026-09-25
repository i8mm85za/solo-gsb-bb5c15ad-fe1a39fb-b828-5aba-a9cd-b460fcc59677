"""窑次温控核验：由离散温度采样求落在合格温区内的最长连续时段。

温度曲线约定：
  * 相邻采样点之间温度按线性变化；
  * 相邻采样间隔超过最大采样间隔时曲线断开（该间隔内不做插值）；
  * 不在首末采样点之外外推。

合格温区为该窑器物温区交集（闭区间）。温度穿越温区边界的时刻一般为
有理数，全部计算使用 Fraction 精确进行，时段起止与时长均以约分分数给出。
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction


def _segment_qualified_interval(
    t0: int,
    temp0: int,
    t1: int,
    temp1: int,
    low: int,
    high: int,
) -> tuple[Fraction, Fraction] | None:
    """线性段 [t0, t1] 上温度落在闭区间 [low, high] 内的闭子区间。

    温度 T(t) = temp0 + (temp1 - temp0) * (t - t0) / (t1 - t0)，关于 t 单调，
    故合格部分必为单个闭区间（可能退化为一点）。返回 (起, 止)（起 <= 止，
    均为精确 Fraction）；本段无合格点时返回 None。
    """

    dt = t1 - t0  # 采样时刻严格递增（调用前已校验），dt > 0
    d_temp = temp1 - temp0
    if d_temp == 0:
        if low <= temp0 <= high:
            return Fraction(t0), Fraction(t1)
        return None
    # 参数 s = (t - t0) / dt ∈ [0, 1]，温度关于 s 单调；
    # low <= T <= high 对应 s 落在 [min(s1, s2), max(s1, s2)]。
    s1 = Fraction(low - temp0, d_temp)
    s2 = Fraction(high - temp0, d_temp)
    s_lo = max(min(s1, s2), Fraction(0))
    s_hi = min(max(s1, s2), Fraction(1))
    if s_lo > s_hi:
        return None
    return t0 + s_lo * dt, t0 + s_hi * dt


def longest_qualified_period(
    samples: Sequence[tuple[int, int]],
    temp_low: int,
    temp_high: int,
    max_interval_seconds: int,
) -> tuple[Fraction | None, Fraction | None, Fraction]:
    """求温度落在合格温区 [temp_low, temp_high] 内的最长连续时段。

    逐段求合格子区间，相邻段在共享采样点处合格区间相接（前段止 == 后段起）
    时合并为同一连续时段；采样间隔超过 max_interval_seconds 时段间断开。

    :param samples: (时刻秒, 温度) 序列，时刻严格递增（调用前已校验）
    :param temp_low: 合格温区下界（该窑器物温区交集下界）
    :param temp_high: 合格温区上界（交集上界）；temp_low > temp_high 视为无合格温区
    :param max_interval_seconds: 最大采样间隔秒数，相邻采样间隔超过它时曲线断开
    :returns: (起始时刻, 结束时刻, 时长)。存在正时长时段时三者均为精确
        Fraction，等长时段取开始最早者；无正时长时段时起止为 None、
        时长为 Fraction(0)。
    """

    best_start: Fraction | None = None
    best_end: Fraction | None = None
    best_duration = Fraction(0)

    if temp_low > temp_high:
        return None, None, best_duration

    def _consider(start: Fraction, end: Fraction) -> None:
        nonlocal best_start, best_end, best_duration
        duration = end - start
        if duration <= 0:
            return  # 退化为点的合格区间不计入（无正时长）
        if duration > best_duration:
            best_start, best_end, best_duration = start, end, duration
        elif (
            duration == best_duration
            and best_start is not None
            and start < best_start
        ):
            best_start, best_end = start, end

    # 当前未结算的连续合格时段 [cur_start, cur_end]
    cur_start: Fraction | None = None
    cur_end: Fraction | None = None

    def _flush() -> None:
        nonlocal cur_start, cur_end
        if cur_start is not None:
            assert cur_end is not None  # 起止总是同时赋值
            _consider(cur_start, cur_end)
            cur_start = cur_end = None

    for i in range(len(samples) - 1):
        t0, temp0 = samples[i]
        t1, temp1 = samples[i + 1]
        if t1 - t0 > max_interval_seconds:
            _flush()  # 采样间隔超限：曲线断开，不做插值
            continue
        interval = _segment_qualified_interval(
            t0, temp0, t1, temp1, temp_low, temp_high
        )
        if interval is None:
            _flush()  # 本段完全在温区外，连续时段中断
            continue
        start, end = interval
        if cur_start is not None and cur_end == start:
            cur_end = end  # 与上一段在共享采样点处相接，并入当前时段
        else:
            _flush()
            cur_start, cur_end = start, end
    _flush()

    return best_start, best_end, best_duration

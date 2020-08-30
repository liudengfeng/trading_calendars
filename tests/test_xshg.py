from trading_calendars import get_calendar
from pytz import timezone
from datetime import time
import pandas as pd
import pytest

tz = timezone("Asia/Shanghai")


@pytest.fixture
def calendar():
    yield get_calendar('XSHG')


@pytest.mark.parametrize("start, end,expected", [
    ('2020-01-07 09:31', '2020-01-07 15:01', 240),
    ('2020-01-07 11:30', '2020-01-07 13:01', 2),
    ('2020-01-07 11:32', '2020-01-07 12:59', 0),
])
def test_minutes_in_range(calendar, start, end, expected):
    start = pd.Timestamp(start, tz=tz).tz_convert('utc')
    end = pd.Timestamp(end, tz=tz).tz_convert('utc')
    actual = calendar.minutes_in_range(start, end)
    assert len(actual) == expected


@pytest.mark.parametrize("dt, expected", [
    ('2020-01-05 11:29', False), # 非交易日
    ('2020-01-06 11:29', True),
    ('2020-01-07 11:29', True),
])
def test_is_session(calendar, dt, expected):
    dt = pd.Timestamp(dt, tz=tz).tz_convert('utc')
    d = dt.normalize()
    assert calendar.is_session(d) == expected


@pytest.mark.parametrize(
    "dt, expected",
    [
        ('2020-01-07 11:29', '2020-01-07 11:30'),
        ('2020-01-07 11:30', '2020-01-07 13:01'),
        ('2020-01-07 11:31', '2020-01-07 13:01'),
        # 跨日
        ('2020-01-06 15:31', '2020-01-07 09:31'),
    ])
def test_next_minute(calendar, dt, expected):
    dt = pd.Timestamp(dt, tz=tz).tz_convert('utc')
    actual = calendar.next_minute(dt).tz_convert(tz)
    assert actual == pd.Timestamp(expected, tz=tz)


@pytest.mark.parametrize(
    "dt, expected",
    [
        ('2020-01-07 11:30', '2020-01-07 11:29'),
        ('2020-01-07 13:00', '2020-01-07 11:30'),
        ('2020-01-07 13:01', '2020-01-07 11:30'),
        # 跨日
        ('2020-01-07 09:31', '2020-01-06 15:00'),
    ])
def test_previous_minute(calendar, dt, expected):
    dt = pd.Timestamp(dt, tz=tz).tz_convert('utc')
    actual = calendar.previous_minute(dt).tz_convert(tz)
    assert actual == pd.Timestamp(expected, tz=tz)

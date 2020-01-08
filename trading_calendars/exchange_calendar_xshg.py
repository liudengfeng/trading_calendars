from datetime import time
from functools import lru_cache

import pandas as pd
from pytz import timezone

from cnswd.reader import calendar
from cnswd.setting.constants import MARKET_START

from .precomputed_trading_calendar import PrecomputedTradingCalendar


@lru_cache()
def get_shanghai_holidays():
    trading_dates = calendar()
    trading_dates = pd.DatetimeIndex(trading_dates)
    all_dates = pd.date_range(MARKET_START.tz_localize(None),
                              pd.Timestamp('today'),
                              freq='D')
    holidays = [x for x in all_dates if x not in trading_dates]
    return holidays


# 如不能正确处理交易日期，会导致fill出现错误
precomputed_shanghai_holidays = pd.to_datetime(
    [x for x in get_shanghai_holidays()])


class XSHGExchangeCalendar(PrecomputedTradingCalendar):
    """
    Exchange calendar for the Shanghai Stock Exchange (XSHG, XSSC, SSE).

    Open time: 9:30 Asia/Shanghai
    Close time: 15:00 Asia/Shanghai

    NOTE: For now, we are skipping the intra-day break from 11:30 to 13:00.

    Due to the complexity around the Shanghai exchange holidays, we are
    hardcoding a list of holidays covering 1999-2025, inclusive. There are
    no known early closes or late opens.
    """

    name = "XSHG"
    tz = timezone("Asia/Shanghai")
    open_times = ((None, time(9, 31)), )
    am_end = ((None, time(11, 30)), )
    # 自1开始，每天240个交易分钟
    pm_start = ((None, time(13, 1)), )
    close_times = ((None, time(15, 0)), )


    @property
    def precomputed_holidays(self):
        return precomputed_shanghai_holidays

    @property
    def actual_last_session(self):
        """实际最后交易日"""
        now = pd.Timestamp.utcnow()
        trading_days = self.all_sessions
        actual = trading_days[trading_days.get_loc(now, method='ffill')]
        return actual

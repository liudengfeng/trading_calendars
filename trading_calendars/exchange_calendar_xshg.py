from datetime import time
from functools import lru_cache

import pandas as pd
from pytz import timezone

# from cnswd.mongodb import get_db
# from cnswd.setting.constants import MARKET_START

from .precomputed_trading_calendar_cn import PrecomputedTradingCalendar
from cnswd.scripts.trading_calendar import get_tdates, is_trading_day

precomputed_shanghai_holidays = pd.to_datetime([
    "1999-01-01",
    "1999-02-10",
    "1999-02-11",
    "1999-02-12",
    "1999-02-15",
    "1999-02-16",
    "1999-02-17",
    "1999-02-18",
    "1999-02-19",
    "1999-02-22",
    "1999-02-23",
    "1999-02-24",
    "1999-02-25",
    "1999-02-26",
    "1999-05-03",
    "1999-10-01",
    "1999-10-04",
    "1999-10-05",
    "1999-10-06",
    "1999-10-07",
    "1999-12-20",
    "1999-12-31",
    "2000-01-03",
    "2000-01-31",
    "2000-02-01",
    "2000-02-02",
    "2000-02-03",
    "2000-02-04",
    "2000-02-07",
    "2000-02-08",
    "2000-02-09",
    "2000-02-10",
    "2000-02-11",
    "2000-05-01",
    "2000-05-02",
    "2000-05-03",
    "2000-05-04",
    "2000-05-05",
    "2000-10-02",
    "2000-10-03",
    "2000-10-04",
    "2000-10-05",
    "2000-10-06",
    "2001-01-01",
    "2001-01-22",
    "2001-01-23",
    "2001-01-24",
    "2001-01-25",
    "2001-01-26",
    "2001-01-29",
    "2001-01-30",
    "2001-01-31",
    "2001-02-01",
    "2001-02-02",
    "2001-05-01",
    "2001-05-02",
    "2001-05-03",
    "2001-05-04",
    "2001-05-07",
    "2001-10-01",
    "2001-10-02",
    "2001-10-03",
    "2001-10-04",
    "2001-10-05",
    "2002-01-01",
    "2002-01-02",
    "2002-01-03",
    "2002-02-11",
    "2002-02-12",
    "2002-02-13",
    "2002-02-14",
    "2002-02-15",
    "2002-02-18",
    "2002-02-19",
    "2002-02-20",
    "2002-02-21",
    "2002-02-22",
    "2002-05-01",
    "2002-05-02",
    "2002-05-03",
    "2002-05-06",
    "2002-05-07",
    "2002-09-30",
    "2002-10-01",
    "2002-10-02",
    "2002-10-03",
    "2002-10-04",
    "2002-10-07",
    "2003-01-01",
    "2003-01-30",
    "2003-01-31",
    "2003-02-03",
    "2003-02-04",
    "2003-02-05",
    "2003-02-06",
    "2003-02-07",
    "2003-05-01",
    "2003-05-02",
    "2003-05-05",
    "2003-05-06",
    "2003-05-07",
    "2003-05-08",
    "2003-05-09",
    "2003-10-01",
    "2003-10-02",
    "2003-10-03",
    "2003-10-06",
    "2003-10-07",
    "2004-01-01",
    "2004-01-19",
    "2004-01-20",
    "2004-01-21",
    "2004-01-22",
    "2004-01-23",
    "2004-01-26",
    "2004-01-27",
    "2004-01-28",
    "2004-05-03",
    "2004-05-04",
    "2004-05-05",
    "2004-05-06",
    "2004-05-07",
    "2004-10-01",
    "2004-10-04",
    "2004-10-05",
    "2004-10-06",
    "2004-10-07",
    "2005-01-03",
    "2005-02-07",
    "2005-02-08",
    "2005-02-09",
    "2005-02-10",
    "2005-02-11",
    "2005-02-14",
    "2005-02-15",
    "2005-05-02",
    "2005-05-03",
    "2005-05-04",
    "2005-05-05",
    "2005-05-06",
    "2005-10-03",
    "2005-10-04",
    "2005-10-05",
    "2005-10-06",
    "2005-10-07",
    "2006-01-02",
    "2006-01-03",
    "2006-01-26",
    "2006-01-27",
    "2006-01-30",
    "2006-01-31",
    "2006-02-01",
    "2006-02-02",
    "2006-02-03",
    "2006-05-01",
    "2006-05-02",
    "2006-05-03",
    "2006-05-04",
    "2006-05-05",
    "2006-10-02",
    "2006-10-03",
    "2006-10-04",
    "2006-10-05",
    "2006-10-06",
    "2007-01-01",
    "2007-01-02",
    "2007-01-03",
    "2007-02-19",
    "2007-02-20",
    "2007-02-21",
    "2007-02-22",
    "2007-02-23",
    "2007-05-01",
    "2007-05-02",
    "2007-05-03",
    "2007-05-04",
    "2007-05-07",
    "2007-10-01",
    "2007-10-02",
    "2007-10-03",
    "2007-10-04",
    "2007-10-05",
    "2007-12-31",
    "2008-01-01",
    "2008-02-06",
    "2008-02-07",
    "2008-02-08",
    "2008-02-11",
    "2008-02-12",
    "2008-04-04",
    "2008-05-01",
    "2008-05-02",
    "2008-06-09",
    "2008-09-15",
    "2008-09-29",
    "2008-09-30",
    "2008-10-01",
    "2008-10-02",
    "2008-10-03",
    "2009-01-01",
    "2009-01-02",
    "2009-01-26",
    "2009-01-27",
    "2009-01-28",
    "2009-01-29",
    "2009-01-30",
    "2009-04-06",
    "2009-05-01",
    "2009-05-28",
    "2009-05-29",
    "2009-10-01",
    "2009-10-02",
    "2009-10-05",
    "2009-10-06",
    "2009-10-07",
    "2009-10-08",
    "2010-01-01",
    "2010-02-15",
    "2010-02-16",
    "2010-02-17",
    "2010-02-18",
    "2010-02-19",
    "2010-04-05",
    "2010-05-03",
    "2010-06-14",
    "2010-06-15",
    "2010-06-16",
    "2010-09-22",
    "2010-09-23",
    "2010-09-24",
    "2010-10-01",
    "2010-10-04",
    "2010-10-05",
    "2010-10-06",
    "2010-10-07",
    "2011-01-03",
    "2011-02-02",
    "2011-02-03",
    "2011-02-04",
    "2011-02-07",
    "2011-02-08",
    "2011-04-04",
    "2011-04-05",
    "2011-05-02",
    "2011-06-06",
    "2011-09-12",
    "2011-10-03",
    "2011-10-04",
    "2011-10-05",
    "2011-10-06",
    "2011-10-07",
    "2012-01-02",
    "2012-01-03",
    "2012-01-23",
    "2012-01-24",
    "2012-01-25",
    "2012-01-26",
    "2012-01-27",
    "2012-04-02",
    "2012-04-03",
    "2012-04-04",
    "2012-04-30",
    "2012-05-01",
    "2012-06-22",
    "2012-10-01",
    "2012-10-02",
    "2012-10-03",
    "2012-10-04",
    "2012-10-05",
    "2013-01-01",
    "2013-01-02",
    "2013-01-03",
    "2013-02-11",
    "2013-02-12",
    "2013-02-13",
    "2013-02-14",
    "2013-02-15",
    "2013-04-04",
    "2013-04-05",
    "2013-04-29",
    "2013-04-30",
    "2013-05-01",
    "2013-06-10",
    "2013-06-11",
    "2013-06-12",
    "2013-09-19",
    "2013-09-20",
    "2013-10-01",
    "2013-10-02",
    "2013-10-03",
    "2013-10-04",
    "2013-10-07",
    "2014-01-01",
    "2014-01-31",
    "2014-02-03",
    "2014-02-04",
    "2014-02-05",
    "2014-02-06",
    "2014-04-07",
    "2014-05-01",
    "2014-05-02",
    "2014-06-02",
    "2014-09-08",
    "2014-10-01",
    "2014-10-02",
    "2014-10-03",
    "2014-10-06",
    "2014-10-07",
    "2015-01-01",
    "2015-01-02",
    "2015-02-18",
    "2015-02-19",
    "2015-02-20",
    "2015-02-23",
    "2015-02-24",
    "2015-04-06",
    "2015-05-01",
    "2015-06-22",
    "2015-09-03",
    "2015-09-04",
    "2015-10-01",
    "2015-10-02",
    "2015-10-05",
    "2015-10-06",
    "2015-10-07",
    "2016-01-01",
    "2016-02-08",
    "2016-02-09",
    "2016-02-10",
    "2016-02-11",
    "2016-02-12",
    "2016-04-04",
    "2016-05-02",
    "2016-06-09",
    "2016-06-10",
    "2016-09-15",
    "2016-09-16",
    "2016-10-03",
    "2016-10-04",
    "2016-10-05",
    "2016-10-06",
    "2016-10-07",
    "2017-01-02",
    "2017-01-27",
    "2017-01-30",
    "2017-01-31",
    "2017-02-01",
    "2017-02-02",
    "2017-04-03",
    "2017-04-04",
    "2017-05-01",
    "2017-05-29",
    "2017-05-30",
    "2017-10-02",
    "2017-10-03",
    "2017-10-04",
    "2017-10-05",
    "2017-10-06",
    "2018-01-01",
    "2018-02-15",
    "2018-02-16",
    "2018-02-19",
    "2018-02-20",
    "2018-02-21",
    "2018-04-05",
    "2018-04-06",
    "2018-04-30",
    "2018-05-01",
    "2018-06-18",
    "2018-09-24",
    "2018-10-01",
    "2018-10-02",
    "2018-10-03",
    "2018-10-04",
    "2018-10-05",
    "2018-12-31",
    "2019-01-01",
    "2019-02-04",
    "2019-02-05",
    "2019-02-06",
    "2019-02-07",
    "2019-02-08",
    "2019-04-05",
    "2019-05-01",
    "2019-05-02",
    "2019-05-03",
    "2019-06-07",
    "2019-09-13",
    "2019-10-01",
    "2019-10-02",
    "2019-10-03",
    "2019-10-04",
    "2019-10-07",
    "2020-01-01",
    "2020-01-24",
    "2020-01-27",
    "2020-01-28",
    "2020-01-29",
    "2020-01-30",
    "2020-04-06",
    "2020-05-01",
    "2020-05-04",
    "2020-05-05",
    "2020-06-25",
    "2020-06-26",
    "2020-10-01",
    "2020-10-02",
    "2020-10-05",
    "2020-10-06",
    "2020-10-07",
    "2020-10-08",
    "2021-01-01",
    "2021-02-11",
    "2021-02-12",
    "2021-02-15",
    "2021-02-16",
    "2021-02-17",
    "2021-04-05",
    "2021-05-03",
    "2021-06-14",
    "2021-09-20",
    "2021-09-21",
    "2021-10-01",
    "2021-10-04",
    "2021-10-05",
    "2021-10-06",
    "2021-10-07",
    "2022-01-03",
    "2022-02-01",
    "2022-02-02",
    "2022-02-03",
    "2022-02-04",
    "2022-04-05",
    "2022-06-03",
    "2022-10-03",
    "2022-10-04",
    "2022-10-05",
    "2023-01-02",
    "2023-01-23",
    "2023-01-24",
    "2023-01-25",
    "2023-01-26",
    "2023-01-27",
    "2023-04-05",
    "2023-05-01",
    "2023-06-22",
    "2023-09-29",
    "2023-10-02",
    "2023-10-03",
    "2023-10-04",
    "2023-10-05",
    "2024-01-01",
    "2024-02-12",
    "2024-02-13",
    "2024-02-14",
    "2024-02-15",
    "2024-04-04",
    "2024-05-01",
    "2024-06-10",
    "2024-09-17",
    "2024-10-01",
    "2024-10-02",
    "2024-10-03",
    "2024-10-04",
    "2025-01-01",
    "2025-01-29",
    "2025-01-30",
    "2025-01-31",
    "2025-02-03",
    "2025-04-04",
    "2025-05-01",
    "2025-10-01",
    "2025-10-02",
    "2025-10-03",
    "2025-10-04",
])


def get_precomputed_shanghai_holidays():
    """取决于后台数据更新时间，某个时点可能存在当日为交易日，但误为假日的情形。"""
    tdates = pd.to_datetime(get_tdates())
    today = pd.Timestamp.today().floor('D')
    cond = precomputed_shanghai_holidays > today
    fholidays = precomputed_shanghai_holidays[cond]
    wdates = pd.date_range(tdates[0], today, freq='B')
    hholidays = wdates.difference(tdates)
    return hholidays.append(fholidays)


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
    # 每天240个交易分钟。
    pm_start = ((None, time(13, 1)), )
    close_times = ((None, time(15, 0)), )

    @property
    def precomputed_holidays(self):
        return get_precomputed_shanghai_holidays()

    @property
    def actual_last_session(self):
        """实际最后交易日"""
        now = pd.Timestamp.utcnow()
        trading_days = self.all_sessions
        actual = trading_days[trading_days.get_loc(now, method='ffill')]
        return actual

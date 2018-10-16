#
# Copyright 2018 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from datetime import time
from pandas.tseries.holiday import (
    DateOffset,
    EasterMonday,
    GoodFriday,
    Holiday,
    MO,
    next_monday,
    next_monday_or_tuesday,
    previous_workday,
    weekend_to_monday,
)
from pytz import timezone

from .common_holidays import (
    new_years_day,
    anzac_day,
    christmas,
    boxing_day,
)

from .trading_calendar import (
    TradingCalendar,
    HolidayCalendar
)


# Prior to 2015, Waitangi Day and Anzac Day are not "Mondayized",
# that is, if they occur on the weekend, there is no make-up.
MONDAYIZATION_START_DATE = "2015-01-01"

# Regular Holidays
# ----------------
NewYearsDay = new_years_day(observance=next_monday)

DayAfterNewYearsDay = Holiday(
    "Day after New Year's Day",
    month=1,
    day=2,
    observance=next_monday_or_tuesday,
)

WaitangiDayNonMondayized = Holiday(
    "Waitangi Day",
    month=2,
    day=6,
    end_date=MONDAYIZATION_START_DATE,
)

WaitangiDay = Holiday(
    "Waitangi Day",
    month=2,
    day=6,
    observance=weekend_to_monday,
    start_date=MONDAYIZATION_START_DATE,
)

AnzacDayNonMondayized = anzac_day(end_date=MONDAYIZATION_START_DATE)

AnzacDay = anzac_day(
    observance=weekend_to_monday,
    start_date=MONDAYIZATION_START_DATE,
)

QueensBirthday = Holiday(
    "Queen's Birthday",
    month=6,
    day=1,
    offset=DateOffset(weekday=MO(1)),
)

LabourDay = Holiday(
    "Labour Day",
    month=10,
    day=1,
    offset=DateOffset(weekday=MO(4)),
)

Christmas = christmas(observance=next_monday)

BoxingDay = boxing_day(observance=next_monday_or_tuesday)


# Early Closes
# ------------
BusinessDayPriorToChristmasDay = Holiday(
    "Business Day prior to Christmas Day",
    month=12,
    day=25,
    observance=previous_workday,
    start_date="2011-01-01",
)

BusinessDayPriorToNewYearsDay = Holiday(
    "Business Day prior to New Year's Day",
    month=1,
    day=1,
    observance=previous_workday,
    start_date="2011-01-01",
)


class XNZEExchangeCalendar(TradingCalendar):
    """
    Exchange calendar for the New Zealand Exchange (NZX).

    Open Time: 10:00 AM, NZ
    Close Time: 4:45 PM, NZ

    Regularly-Observed Holidays:
    - New Year's Day
    - Day after New Year's Day
    - Waitangi Day
    - Good Friday
    - Easter Monday
    - Anzac Day
    - Queen's Birthday
    - Labour Day
    - Christmas
    - Boxing Day

    Early Closes:
    - Business Day prior to Christmas Day
    - Business Day prior to New Year's Day
    """
    regular_early_close = time(12, 45)

    @property
    def name(self):
        return "XNZE"

    @property
    def tz(self):
        return timezone('NZ')

    @property
    def open_time(self):
        return time(10, 1)

    @property
    def close_time(self):
        return time(16, 45)

    @property
    def regular_holidays(self):
        return HolidayCalendar([
            NewYearsDay,
            DayAfterNewYearsDay,
            WaitangiDayNonMondayized,
            WaitangiDay,
            GoodFriday,
            EasterMonday,
            AnzacDayNonMondayized,
            AnzacDay,
            QueensBirthday,
            LabourDay,
            Christmas,
            BoxingDay,
        ])

    @property
    def special_closes(self):
        return [
            (
                self.regular_early_close,
                HolidayCalendar([
                    BusinessDayPriorToChristmasDay,
                    BusinessDayPriorToNewYearsDay
                ])
            )
        ]
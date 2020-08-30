from os.path import abspath, dirname, join
from unittest import TestCase

import numpy as np
import pandas as pd
from nose_parameterized import parameterized
from pandas import Timedelta, read_csv
from pandas.util.testing import assert_index_equal
from pytz import UTC

from trading_calendars import all_trading_minutes, get_calendar
from trading_calendars.exchange_calendar_xshg import XSHGExchangeCalendar


def T(x):
    return pd.Timestamp(x, tz=UTC)


class XSHGCalendarTestCase(TestCase):

    # Override in subclasses.
    answer_key_filename = 'xshg'
    calendar_class = XSHGExchangeCalendar

    # Affects tests that care about the empty periods between sessions. Should
    # be set to False for 24/7 calendars.
    GAPS_BETWEEN_SESSIONS = True

    # Affects tests that care about early closes. Should be set to False for
    # calendars that don't have any early closes.
    HAVE_EARLY_CLOSES = True

    # Affects tests that care about late opens. Since most do not, defaulting
    # to False.
    HAVE_LATE_OPENS = False

    # Affects test_sanity_check_session_lengths. Should be set to the largest
    # number of hours that ever appear in a single session.
    MAX_SESSION_HOURS = 0

    # Affects test_minute_index_to_session_labels.
    # Change these if the start/end dates of your test suite don't contain the
    # defaults.
    MINUTE_INDEX_TO_SESSION_LABELS_START = pd.Timestamp('2011-01-04', tz=UTC)
    MINUTE_INDEX_TO_SESSION_LABELS_END = pd.Timestamp('2011-04-04', tz=UTC)

    # Affects tests around daylight savings. If possible, should contain two
    # dates that are not both in the same daylight savings regime.
    DAYLIGHT_SAVINGS_DATES = ["2004-04-05", "2004-11-01"]

    # Affects test_start_end. Change these if your calendar start/end
    # dates between 2010-01-03 and 2010-01-10 don't match the defaults.
    TEST_START_END_FIRST = pd.Timestamp('2010-01-03', tz=UTC)
    TEST_START_END_LAST = pd.Timestamp('2010-01-10', tz=UTC)
    TEST_START_END_EXPECTED_FIRST = pd.Timestamp('2010-01-04', tz=UTC)
    TEST_START_END_EXPECTED_LAST = pd.Timestamp('2010-01-08', tz=UTC)

    MAX_SESSION_HOURS = 5.5
    HALF_SESSION_HOURS = 2.0

    HAVE_EARLY_CLOSES = False

    MINUTE_INDEX_TO_SESSION_LABELS_END = pd.Timestamp('2011-04-07', tz=UTC)

    @staticmethod
    def load_answer_key(filename):
        """
        Load a CSV from tests/resources/{filename}.csv
        """
        fullpath = join(
            dirname(abspath(__file__)),
            './resources',
            filename + '.csv',
        )

        return read_csv(
            fullpath,
            index_col=0,
            # NOTE: Merely passing parse_dates=True doesn't cause pandas to set
            # the dtype correctly, and passing all reasonable inputs to the
            # dtype kwarg cause read_csv to barf.
            parse_dates=[0, 1, 2],
            date_parser=lambda x: pd.Timestamp(x, tz=UTC)
        )

    def setUp(self):
        self.answers = self.load_answer_key(self.answer_key_filename)

        self.start_date = self.answers.index[0]
        self.end_date = self.answers.index[-1]
        self.calendar = self.calendar_class(self.start_date, self.end_date)

        self.one_minute = pd.Timedelta(minutes=1)
        self.one_hour = pd.Timedelta(hours=1)

    def tearDown(self):
        self.calendar = None
        self.answers = None

    def test_sanity_check_session_lengths(self):
        # make sure that no session is longer than self.MAX_SESSION_HOURS hours
        for session in self.calendar.all_sessions:
            o, c = self.calendar.open_and_close_for_session(session)
            delta = c - o
            self.assertLessEqual(delta.seconds / 3600, self.MAX_SESSION_HOURS)

    def test_sanity_check_am_session_lengths(self):
        # make sure that no session is longer than self.HALF_SESSION_HOURS hours
        for session in self.calendar.all_sessions:
            o, c = self.calendar.am_open_and_close_for_session(session)
            delta = c - o
            self.assertLessEqual(delta.seconds / 3600, self.HALF_SESSION_HOURS)

    def test_sanity_check_pm_session_lengths(self):
        # make sure that no session is longer than self.HALF_SESSION_HOURS hours
        for session in self.calendar.all_sessions:
            o, c = self.calendar.pm_open_and_close_for_session(session)
            delta = c - o
            self.assertLessEqual(delta.seconds / 3600, self.HALF_SESSION_HOURS)

    def test_calculated_against_csv(self):
        assert_index_equal(self.calendar.schedule.index, self.answers.index)

    def test_is_open_on_minute(self):
        one_minute = pd.Timedelta(minutes=1)

        for market_minute in self.answers.market_open:
            market_minute_utc = market_minute
            # The exchange should be classified as open on its first minute
            self.assertTrue(self.calendar.is_open_on_minute(market_minute_utc))

            if self.GAPS_BETWEEN_SESSIONS:
                # Decrement minute by one, to minute where the market was not
                # open
                pre_market = market_minute_utc - one_minute
                self.assertFalse(self.calendar.is_open_on_minute(pre_market))

        for market_minute in self.answers.market_close:
            close_minute_utc = market_minute
            # should be open on its last minute
            self.assertTrue(self.calendar.is_open_on_minute(close_minute_utc))

            if self.GAPS_BETWEEN_SESSIONS:
                # increment minute by one minute, should be closed
                post_market = close_minute_utc + one_minute
                self.assertFalse(self.calendar.is_open_on_minute(post_market))

    def _verify_minute(self, calendar, minute,
                       next_open_answer, prev_open_answer,
                       next_close_answer, prev_close_answer):
        self.assertEqual(
            calendar.next_open(minute),
            next_open_answer
        )

        self.assertEqual(
            self.calendar.previous_open(minute),
            prev_open_answer
        )

        self.assertEqual(
            self.calendar.next_close(minute),
            next_close_answer
        )

        self.assertEqual(
            self.calendar.previous_close(minute),
            prev_close_answer
        )

    def test_next_prev_open_close(self):
        # for each session, check:
        # - the minute before the open (if gaps exist between sessions)
        # - the first minute of the session
        # - the second minute of the session
        # - the minute before the close
        # - the last minute of the session
        # - the first minute after the close (if gaps exist between sessions)
        opens = self.answers.market_open.iloc[1:-2]
        closes = self.answers.market_close.iloc[1:-2]

        previous_opens = self.answers.market_open.iloc[:-1]
        previous_closes = self.answers.market_close.iloc[:-1]

        next_opens = self.answers.market_open.iloc[2:]
        next_closes = self.answers.market_close.iloc[2:]

        for (open_minute, close_minute,
             previous_open, previous_close,
             next_open, next_close) in zip(opens, closes,
                                           previous_opens, previous_closes,
                                           next_opens, next_closes):

            minute_before_open = open_minute - self.one_minute

            # minute before open
            if self.GAPS_BETWEEN_SESSIONS:
                self._verify_minute(
                    self.calendar, minute_before_open, open_minute,
                    previous_open, close_minute, previous_close
                )

            # open minute
            self._verify_minute(
                self.calendar, open_minute, next_open, previous_open,
                close_minute, previous_close
            )

            # second minute of session
            self._verify_minute(
                self.calendar, open_minute + self.one_minute, next_open,
                open_minute, close_minute, previous_close
            )

            # minute before the close
            self._verify_minute(
                self.calendar, close_minute - self.one_minute, next_open,
                open_minute, close_minute, previous_close
            )

            # the close
            self._verify_minute(
                self.calendar, close_minute, next_open, open_minute,
                next_close, previous_close
            )

            # minute after the close
            if self.GAPS_BETWEEN_SESSIONS:
                self._verify_minute(
                    self.calendar, close_minute + self.one_minute, next_open,
                    open_minute, next_close, close_minute
                )

    def test_next_prev_minute(self):
        all_minutes = self.calendar.all_minutes

        # test 20,000 minutes because it takes too long to do the rest.
        for idx, minute in enumerate(all_minutes[1:20000]):
            self.assertEqual(
                all_minutes[idx + 2],
                self.calendar.next_minute(minute)
            )

            self.assertEqual(
                all_minutes[idx],
                self.calendar.previous_minute(minute)
            )

        # test a couple of non-market minutes
        if self.GAPS_BETWEEN_SESSIONS:
            for open_minute in self.answers.market_open[1:]:
                hour_before_open = open_minute - self.one_hour
                self.assertEqual(
                    open_minute,
                    self.calendar.next_minute(hour_before_open)
                )

            for close_minute in self.answers.market_close[1:]:
                hour_after_close = close_minute + self.one_hour
                self.assertEqual(
                    close_minute,
                    self.calendar.previous_minute(hour_after_close)
                )

    def test_minute_to_session_label(self):
        for idx, info in enumerate(self.answers[1:-2].iterrows()):
            session_label = info[1].name
            open_minute = info[1].iloc[0]
            close_minute = info[1].iloc[1]
            hour_into_session = open_minute + self.one_hour

            minute_before_session = open_minute - self.one_minute
            minute_after_session = close_minute + self.one_minute

            next_session_label = self.answers.iloc[idx + 2].name
            previous_session_label = self.answers.iloc[idx].name

            # verify that minutes inside a session resolve correctly
            minutes_that_resolve_to_this_session = [
                self.calendar.minute_to_session_label(open_minute),
                self.calendar.minute_to_session_label(open_minute,
                                                      direction="next"),
                self.calendar.minute_to_session_label(open_minute,
                                                      direction="previous"),
                self.calendar.minute_to_session_label(open_minute,
                                                      direction="none"),
                self.calendar.minute_to_session_label(hour_into_session),
                self.calendar.minute_to_session_label(hour_into_session,
                                                      direction="next"),
                self.calendar.minute_to_session_label(hour_into_session,
                                                      direction="previous"),
                self.calendar.minute_to_session_label(hour_into_session,
                                                      direction="none"),
                self.calendar.minute_to_session_label(close_minute),
                self.calendar.minute_to_session_label(close_minute,
                                                      direction="next"),
                self.calendar.minute_to_session_label(close_minute,
                                                      direction="previous"),
                self.calendar.minute_to_session_label(close_minute,
                                                      direction="none"),
                session_label
            ]

            if self.GAPS_BETWEEN_SESSIONS:
                minutes_that_resolve_to_this_session.append(
                    self.calendar.minute_to_session_label(
                        minute_before_session
                    )
                )
                minutes_that_resolve_to_this_session.append(
                    self.calendar.minute_to_session_label(
                        minute_before_session,
                        direction="next"
                    )
                )

                minutes_that_resolve_to_this_session.append(
                    self.calendar.minute_to_session_label(
                        minute_after_session,
                        direction="previous"
                    )
                )

            self.assertTrue(all(x == minutes_that_resolve_to_this_session[0]
                                for x in minutes_that_resolve_to_this_session))

            minutes_that_resolve_to_next_session = [
                self.calendar.minute_to_session_label(minute_after_session),
                self.calendar.minute_to_session_label(minute_after_session,
                                                      direction="next"),
                next_session_label
            ]

            self.assertTrue(all(x == minutes_that_resolve_to_next_session[0]
                                for x in minutes_that_resolve_to_next_session))

            self.assertEqual(
                self.calendar.minute_to_session_label(minute_before_session,
                                                      direction="previous"),
                previous_session_label
            )

            if self.GAPS_BETWEEN_SESSIONS:
                # Make sure we use the cache correctly
                minutes_that_resolve_to_different_sessions = [
                    self.calendar.minute_to_session_label(minute_after_session,
                                                          direction="next"),
                    self.calendar.minute_to_session_label(
                        minute_after_session,
                        direction="previous"
                    ),
                    self.calendar.minute_to_session_label(minute_after_session,
                                                          direction="next"),
                ]

                self.assertEqual(
                    minutes_that_resolve_to_different_sessions,
                    [next_session_label,
                     session_label,
                     next_session_label]
                )

            # make sure that exceptions are raised at the right time
            with self.assertRaises(ValueError):
                self.calendar.minute_to_session_label(open_minute, "asdf")

            if self.GAPS_BETWEEN_SESSIONS:
                with self.assertRaises(ValueError):
                    self.calendar.minute_to_session_label(
                        minute_before_session,
                        direction="none"
                    )

    @parameterized.expand([
        (1, 0),
        (2, 0),
        (2, 1),
    ])
    def test_minute_index_to_session_labels(self, interval, offset):
        minutes = self.calendar.minutes_for_sessions_in_range(
            self.MINUTE_INDEX_TO_SESSION_LABELS_START,
            self.MINUTE_INDEX_TO_SESSION_LABELS_END,
        )
        minutes = minutes[range(offset, len(minutes), interval)]

        np.testing.assert_array_equal(
            pd.DatetimeIndex(
                minutes.map(self.calendar.minute_to_session_label)
            ),
            self.calendar.minute_index_to_session_labels(minutes),
        )

    def test_next_prev_session(self):
        session_labels = self.answers.index[1:-2]
        max_idx = len(session_labels) - 1

        # the very first session
        first_session_label = self.answers.index[0]
        with self.assertRaises(ValueError):
            self.calendar.previous_session_label(first_session_label)

        # all the sessions in the middle
        for idx, session_label in enumerate(session_labels):
            if idx < max_idx:
                self.assertEqual(
                    self.calendar.next_session_label(session_label),
                    session_labels[idx + 1]
                )

            if idx > 0:
                self.assertEqual(
                    self.calendar.previous_session_label(session_label),
                    session_labels[idx - 1]
                )

        # the very last session
        last_session_label = self.answers.index[-1]
        with self.assertRaises(ValueError):
            self.calendar.next_session_label(last_session_label)

    @staticmethod
    def _find_full_session(calendar):
        for session_label in calendar.schedule.index:
            if session_label not in calendar.early_closes:
                return session_label

        return None

    def test_minutes_for_period(self):
        # full session
        # find a session that isn't an early close.  start from the first
        # session, should be quick.
        full_session_label = self._find_full_session(self.calendar)
        if full_session_label is None:
            raise ValueError("Cannot find a full session to test!")

        minutes = self.calendar.minutes_for_session(full_session_label)
        am_open, am_close = self.calendar.am_open_and_close_for_session(
            full_session_label
        )
        pm_open, pm_close = self.calendar.pm_open_and_close_for_session(
            full_session_label
        )

        np.testing.assert_array_equal(
            minutes,
            all_trading_minutes(am_open, pm_close)
        )

        # early close period
        if self.HAVE_EARLY_CLOSES:
            early_close_session_label = self.calendar.early_closes[0]
            minutes_for_early_close = \
                self.calendar.minutes_for_session(early_close_session_label)
            _open, _close = self.calendar.open_and_close_for_session(
                early_close_session_label
            )

            np.testing.assert_array_equal(
                minutes_for_early_close,
                pd.date_range(start=_open, end=_close, freq="min")
            )

        # late open period
        if self.HAVE_LATE_OPENS:
            late_open_session_label = self.calendar.late_opens[0]
            minutes_for_late_open = \
                self.calendar.minutes_for_session(late_open_session_label)
            _open, _close = self.calendar.open_and_close_for_session(
                late_open_session_label
            )

            np.testing.assert_array_equal(
                minutes_for_late_open,
                pd.date_range(start=_open, end=_close, freq="min")
            )

    def test_sessions_in_range(self):
        # pick two sessions
        session_count = len(self.calendar.schedule.index)

        first_idx = session_count // 3
        second_idx = 2 * first_idx

        first_session_label = self.calendar.schedule.index[first_idx]
        second_session_label = self.calendar.schedule.index[second_idx]

        answer_key = \
            self.calendar.schedule.index[first_idx:second_idx + 1]

        np.testing.assert_array_equal(
            answer_key,
            self.calendar.sessions_in_range(first_session_label,
                                            second_session_label)
        )

    def get_session_block(self):
        """
        Get an "interesting" range of three sessions in a row. By default this
        tries to find and return a (full session, early close session, full
        session) block.
        """
        if not self.HAVE_EARLY_CLOSES:
            # If we don't have any early closes, just return a "random" chunk
            # of three sessions.
            return self.calendar.all_sessions[10:13]

        shortened_session = self.calendar.early_closes[0]
        shortened_session_idx = \
            self.calendar.schedule.index.get_loc(shortened_session)

        session_before = self.calendar.schedule.index[
            shortened_session_idx - 1
        ]
        session_after = self.calendar.schedule.index[shortened_session_idx + 1]

        return [session_before, shortened_session, session_after]

    def test_minutes_in_range(self):
        sessions = self.get_session_block()

        first_open, first_close = self.calendar.open_and_close_for_session(
            sessions[0]
        )
        minute_before_first_open = first_open - self.one_minute

        middle_open, middle_close = \
            self.calendar.open_and_close_for_session(sessions[1])

        last_open, last_close = self.calendar.open_and_close_for_session(
            sessions[-1]
        )
        minute_after_last_close = last_close + self.one_minute

        # get all the minutes between first_open and last_close
        minutes1 = self.calendar.minutes_in_range(
            first_open,
            last_close
        )
        # 尽管区间前后各增加1分钟，区间交易分钟应该依然一致
        minutes2 = self.calendar.minutes_in_range(
            minute_before_first_open,
            minute_after_last_close
        )

        if self.GAPS_BETWEEN_SESSIONS:
            np.testing.assert_array_equal(minutes1, minutes2)
        else:
            # if no gaps, then minutes2 should have 2 extra minutes
            np.testing.assert_array_equal(minutes1, minutes2[1:-1])

        # manually construct the minutes
        all_minutes = np.concatenate([
            all_trading_minutes(first_open, first_close),
            all_trading_minutes(middle_open, middle_close),
            all_trading_minutes(last_open, last_close),
        ])

        np.testing.assert_array_equal(all_minutes, minutes1)

    def test_minutes_for_sessions_in_range(self):
        sessions = self.get_session_block()

        minutes = self.calendar.minutes_for_sessions_in_range(
            sessions[0],
            sessions[-1]
        )

        # do it manually
        session0_minutes = self.calendar.minutes_for_session(sessions[0])
        session1_minutes = self.calendar.minutes_for_session(sessions[1])
        session2_minutes = self.calendar.minutes_for_session(sessions[2])

        concatenated_minutes = np.concatenate([
            session0_minutes.values,
            session1_minutes.values,
            session2_minutes.values
        ])

        np.testing.assert_array_equal(
            concatenated_minutes,
            minutes.values
        )

    def test_sessions_window(self):
        sessions = self.get_session_block()

        np.testing.assert_array_equal(
            self.calendar.sessions_window(sessions[0], len(sessions) - 1),
            self.calendar.sessions_in_range(sessions[0], sessions[-1])
        )

        np.testing.assert_array_equal(
            self.calendar.sessions_window(
                sessions[-1],
                -1 * (len(sessions) - 1)),
            self.calendar.sessions_in_range(sessions[0], sessions[-1])
        )

    def test_session_distance(self):
        sessions = self.get_session_block()

        forward_distance = self.calendar.session_distance(
            sessions[0],
            sessions[-1],
        )
        self.assertEqual(forward_distance, len(sessions))

        backward_distance = self.calendar.session_distance(
            sessions[-1],
            sessions[0],
        )
        self.assertEqual(backward_distance, -len(sessions))

        one_day_distance = self.calendar.session_distance(
            sessions[0],
            sessions[0],
        )
        self.assertEqual(one_day_distance, 1)

    def test_open_and_close_for_session(self):
        for index, row in self.answers.iterrows():
            session_label = row.name
            open_answer = row.iloc[0]
            close_answer = row.iloc[1]

            found_open, found_close = \
                self.calendar.open_and_close_for_session(session_label)

            # Test that the methods for just session open and close produce the
            # same values as the method for getting both.
            alt_open = self.calendar.session_open(session_label)
            self.assertEqual(alt_open, found_open)

            alt_close = self.calendar.session_close(session_label)
            self.assertEqual(alt_close, found_close)

            self.assertEqual(open_answer, found_open)
            self.assertEqual(close_answer, found_close)

    def test_session_opens_in_range(self):
        found_opens = self.calendar.session_opens_in_range(
            self.answers.index[0],
            self.answers.index[-1],
        )
        # 🆗 读取数据并不包含freq属性，改用值相等判断
        np.testing.assert_array_equal(
            found_opens.values, self.answers['market_open'].values
        )

    def test_session_closes_in_range(self):
        found_closes = self.calendar.session_closes_in_range(
            self.answers.index[0],
            self.answers.index[-1],
        )
        # 🆗 读取数据并不包含freq属性，改用值相等判断
        np.testing.assert_array_equal(
            found_closes.values, self.answers['market_close'].values
        )

    def test_daylight_savings(self):
        # 2004 daylight savings switches:
        # Sunday 2004-04-04 and Sunday 2004-10-31

        # make sure there's no weirdness around calculating the next day's
        # session's open time.

        m = dict(self.calendar.open_times)
        m[pd.Timestamp.min] = m.pop(None)
        open_times = pd.Series(m)

        for date in self.DAYLIGHT_SAVINGS_DATES:
            next_day = pd.Timestamp(date, tz=UTC)
            open_date = next_day + Timedelta(days=self.calendar.open_offset)

            the_open = self.calendar.schedule.loc[next_day].market_open

            localized_open = the_open.tz_localize(UTC).tz_convert(
                self.calendar.tz
            )

            self.assertEqual(
                (open_date.year, open_date.month, open_date.day),
                (localized_open.year, localized_open.month, localized_open.day)
            )

            open_ix = open_times.index.searchsorted(pd.Timestamp(date),
                                                    side='r')
            if open_ix == len(open_times):
                open_ix -= 1

            self.assertEqual(
                open_times.iloc[open_ix].hour,
                localized_open.hour
            )

            self.assertEqual(
                open_times.iloc[open_ix].minute,
                localized_open.minute
            )

    def test_start_end(self):
        """
        Check TradingCalendar with defined start/end dates.
        """
        calendar = self.calendar_class(
            start=self.TEST_START_END_FIRST,
            end=self.TEST_START_END_LAST,
        )

        self.assertEqual(
            calendar.first_trading_session,
            self.TEST_START_END_EXPECTED_FIRST,
        )
        self.assertEqual(
            calendar.last_trading_session,
            self.TEST_START_END_EXPECTED_LAST,
        )

    def test_normal_year(self):
        expected_holidays_2017 = [
            T("2017-01-02"),
            T("2017-01-27"),
            T("2017-01-30"),
            T("2017-01-31"),
            T("2017-02-01"),
            T("2017-02-02"),
            T("2017-04-03"),
            T("2017-04-04"),
            T("2017-05-01"),
            T("2017-05-29"),
            T("2017-05-30"),
            T("2017-10-02"),
            T("2017-10-03"),
            T("2017-10-04"),
            T("2017-10-05"),
            T("2017-10-06"),
        ]

        for session_label in expected_holidays_2017:
            self.assertNotIn(session_label, self.calendar.all_sessions)

    def test_constrain_construction_dates(self):
        # the XSHG calendar currently goes from 1999 to 2025, inclusive.
        with self.assertRaises(ValueError) as e:
            self.calendar_class(T('1998-12-31'), T('2005-01-01'))

        self.assertEqual(
            str(e.exception),
            (
                'The XSHG holidays are only recorded back to 1999,'
                ' cannot instantiate the XSHG calendar back to 1998.'
            )
        )

        with self.assertRaises(ValueError) as e:
            self.calendar_class(T('2005-01-01'), T('2026-01-01'))

        self.assertEqual(
            str(e.exception),
            (
                'The XSHG holidays are only recorded to 2025,'
                ' cannot instantiate the XSHG calendar for 2026.'
            )
        )

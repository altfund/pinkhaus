#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 02:19:02 2025

@author: ess
"""

from datetime import datetime, timezone
from evaluate_open_markets import *

generate_betting_session_report_and_save(
            execution_bankroll=100,
            avg_game_duration_minutes=200.0,
            min_break_minutes=60.0,
            abs_game_limit=None,
            as_of=datetime(2025, 4, 15, 12, 0, tzinfo=timezone.utc)
            )

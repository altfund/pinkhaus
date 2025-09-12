#!/usr/bin/env python3
"""Debug game chunks and market availability."""

from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market
import pandas as pd
from evaluate_open_markets import (
    summarize_match_schedule_from_open_markets,
    find_upcoming_game_breaks,
    extract_active_game_periods_from_breaks
)

def debug_chunks():
    """Debug why no markets are found in chunks."""
    with db_manager.get_db_session() as db:
        now_utc = datetime.now(timezone.utc)
        print(f"Current time (UTC): {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get all upcoming markets
        all_active_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > now_utc
        ).order_by(Market.maturity_date).limit(20).all()
        
        print(f"\nNext 20 upcoming soccer matches:")
        for market in all_active_markets[:10]:
            time_until = market.maturity_date - now_utc
            hours = time_until.total_seconds() / 3600
            print(f"  {market.home_team} vs {market.away_team} - {market.maturity_date.strftime('%Y-%m-%d %H:%M')} UTC ({hours:.1f}h from now)")
        
        if not all_active_markets:
            print("No active markets found!")
            return
            
        # Convert to DataFrame for chunk calculation
        temp_data = []
        for market in all_active_markets:
            temp_data.append({
                'source_id': market.source_id,
                'maturity_date': market.maturity_date,
                'home_team': market.home_team,
                'away_team': market.away_team,
                'league_name': market.league_name
            })
        temp_df = pd.DataFrame(temp_data)
        
        # Calculate chunks
        match_df = summarize_match_schedule_from_open_markets(temp_df)
        if match_df.empty:
            print("No matches found for chunk calculation")
            return
            
        # Try different parameters
        min_break_minutes = 240  # 4 hours
        avg_game_duration_minutes = 180  # 3 hours
        
        print(f"\nChunk parameters:")
        print(f"  Min break: {min_break_minutes} minutes")
        print(f"  Avg game duration: {avg_game_duration_minutes} minutes")
        
        breaks_df = find_upcoming_game_breaks(
            match_df, 
            min_break_minutes=min_break_minutes,
            avg_game_duration_minutes=avg_game_duration_minutes,
            now=now_utc
        )
        
        print(f"\nGame breaks found: {len(breaks_df)}")
        if not breaks_df.empty:
            print(f"  Break columns: {breaks_df.columns.tolist()}")
            for _, brk in breaks_df.iterrows():
                if 'break_minutes' in brk:
                    print(f"  Break: {brk['break_start'].strftime('%H:%M')} - {brk['break_end'].strftime('%H:%M')} ({brk['break_minutes']} min)")
                else:
                    print(f"  Break: {brk['break_start'].strftime('%H:%M')} - {brk['break_end'].strftime('%H:%M')}")
        
        game_chunks = extract_active_game_periods_from_breaks(
            match_df, breaks_df, 
            avg_game_duration_minutes=avg_game_duration_minutes
        )
        
        print(f"\nGame chunks: {len(game_chunks)}")
        if not game_chunks.empty:
            for _, chunk in game_chunks.iterrows():
                chunk_start = pd.to_datetime(chunk['chunk_start'], utc=True)
                chunk_end = pd.to_datetime(chunk['chunk_end'], utc=True)
                print(f"\nChunk: {chunk_start.strftime('%Y-%m-%d %H:%M')} to {chunk_end.strftime('%Y-%m-%d %H:%M')} ({chunk['num_games']} games)")
                
                # Check markets in this chunk with in-play buffer
                in_play_buffer_minutes = 15
                now_plus_buffer = now_utc + pd.Timedelta(minutes=in_play_buffer_minutes)
                
                chunk_markets = db.query(Market).filter(
                    Market.sport == 'Soccer',
                    Market.is_finished == False,
                    Market.maturity_date >= now_plus_buffer,
                    Market.maturity_date >= chunk_start,
                    Market.maturity_date < chunk_end
                ).all()
                
                print(f"  Markets in chunk (with {in_play_buffer_minutes}min buffer): {len(chunk_markets)}")
                
                # Also check without buffer
                chunk_markets_no_buffer = db.query(Market).filter(
                    Market.sport == 'Soccer',
                    Market.is_finished == False,
                    Market.maturity_date >= chunk_start,
                    Market.maturity_date < chunk_end
                ).all()
                
                print(f"  Markets in chunk (no buffer): {len(chunk_markets_no_buffer)}")
                
                if len(chunk_markets_no_buffer) > len(chunk_markets):
                    print(f"  In-play buffer excluded {len(chunk_markets_no_buffer) - len(chunk_markets)} markets")

if __name__ == "__main__":
    debug_chunks()
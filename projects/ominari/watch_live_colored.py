#!/usr/bin/env python3
"""
Colored live log monitor for Ominari system
Shows chunks, trades, and system activity with colors
"""
import subprocess
import sys
import re
from colorama import init, Fore, Back, Style

# Initialize colorama for cross-platform color support
init(autoreset=True)

def colorize_line(line):
    """Apply color formatting to log lines"""
    
    # Chunk creation
    if "Created" in line and "time-based chunks" in line:
        return f"{Fore.CYAN}{Style.BRIGHT}📅 CHUNKS: {Style.RESET_ALL}{Fore.CYAN}{line}"
    
    # Chunk details
    elif "Chunk" in line and "markets" in line and ":" in line:
        return f"{Fore.CYAN}   └─ {line}"
    
    # Selected chunk for trading
    elif "Selected first chunk" in line:
        return f"{Fore.GREEN}{Style.BRIGHT}🎯 ACTIVE: {Style.RESET_ALL}{Fore.GREEN}{line}"
    
    # Trade execution
    elif "Paper trading cycle complete" in line:
        return f"{Fore.YELLOW}{Style.BRIGHT}💰 TRADES: {Style.RESET_ALL}{Fore.YELLOW}{line}"
    
    # Odds distribution
    elif "Odds distribution" in line:
        return f"{Fore.MAGENTA}📊 ODDS: {Style.RESET_ALL}{Fore.MAGENTA}{line}"
    
    # Sample odds
    elif "Sample odds" in line:
        return f"{Fore.MAGENTA}   └─ {line}"
    
    # Recorded trades
    elif "Recorded" in line and "trades" in line:
        return f"{Fore.GREEN}✅ {line}"
    
    # Error
    elif "ERROR" in line or "error" in line:
        return f"{Fore.RED}{Style.BRIGHT}❌ ERROR: {Style.RESET_ALL}{Fore.RED}{line}"
    
    # Warning
    elif "WARNING" in line or "warning" in line:
        return f"{Fore.YELLOW}⚠️  WARNING: {Style.RESET_ALL}{Fore.YELLOW}{line}"
    
    # Client connections
    elif "Client connected" in line:
        return f"{Fore.BLUE}🔌 CONNECT: {Style.RESET_ALL}{Fore.BLUE}{line}"
    
    # Found markets
    elif "Found" in line and "markets for" in line:
        return f"{Fore.GREEN}✅ MARKETS: {Style.RESET_ALL}{Fore.GREEN}{line}"
    
    # Query returned
    elif "Query returned" in line:
        return f"{Style.DIM}   📊 {line}{Style.RESET_ALL}"
    
    # Calculating edges
    elif "Calculating edges" in line:
        return f"{Style.DIM}   📐 {line}{Style.RESET_ALL}"
    
    # Successfully created signals
    elif "Successfully created" in line and "signals" in line:
        return f"{Style.DIM}   ✓ {line}{Style.RESET_ALL}"
    
    # Default - show dimmed
    else:
        return f"{Style.DIM}{line}{Style.RESET_ALL}"

def main():
    print(f"{Fore.CYAN}{Style.BRIGHT}🎯 OMINARI LIVE SYSTEM MONITOR{Style.RESET_ALL}")
    print("=" * 60)
    print(f"Watching for: {Fore.CYAN}Chunks{Fore.RESET} | {Fore.YELLOW}Trades{Fore.RESET} | {Fore.MAGENTA}Odds{Fore.RESET} | {Fore.RED}Errors{Fore.RESET}")
    print("Press Ctrl+C to exit")
    print("")
    
    try:
        # Use subprocess to tail the log file
        process = subprocess.Popen(
            ['tail', '-f', 'web_monitor_fixed.log'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Process each line as it comes
        for line in process.stdout:
            line = line.strip()
            if line:
                print(colorize_line(line))
                
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Stopping monitor...{Fore.RESET}")
        process.terminate()
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Fore.RESET}")

if __name__ == "__main__":
    main()
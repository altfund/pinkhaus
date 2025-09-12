#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ominari System Daemon
Manages the Ominari trading system as a background service.
"""

import os
import sys
import time
import signal
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timezone
import psutil
import json

# Configure logging
LOG_FILE = Path(__file__).parent / "ominari_daemon.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# PID file for daemon management
PID_FILE = Path(__file__).parent / "ominari_daemon.pid"
STATUS_FILE = Path(__file__).parent / "ominari_status.json"


class OminariDaemon:
    """Daemon to manage the Ominari trading system."""
    
    def __init__(self):
        self.running = False
        self.ominari_process = None
        self.start_time = None
        
    def start(self):
        """Start the daemon."""
        # Check if already running
        if self._is_running():
            logger.warning("Ominari daemon already running")
            return False
            
        # Write PID file
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))
            
        logger.info("Starting Ominari daemon")
        self.running = True
        self.start_time = datetime.now(timezone.utc)
        
        # Set up signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Start the main loop
        self._run()
        
        return True
        
    def stop(self):
        """Stop the daemon."""
        logger.info("Stopping Ominari daemon")
        self.running = False
        
        # Stop Ominari system
        if self.ominari_process:
            self._stop_ominari_system()
            
        # Remove PID file
        if PID_FILE.exists():
            PID_FILE.unlink()
            
        # Update status
        self._update_status("stopped")
        
    def _signal_handler(self, signum, frame):
        """Handle system signals."""
        logger.info(f"Received signal {signum}")
        self.stop()
        
    def _is_running(self):
        """Check if daemon is already running."""
        if not PID_FILE.exists():
            return False
            
        try:
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
                
            # Check if process exists
            if psutil.pid_exists(pid):
                return True
            else:
                # Stale PID file
                PID_FILE.unlink()
                return False
        except Exception:
            return False
            
    def _run(self):
        """Main daemon loop."""
        while self.running:
            try:
                # Check if Ominari system is running
                if not self._is_ominari_running():
                    logger.info("Starting Ominari system")
                    self._start_ominari_system()
                else:
                    # Check health
                    self._check_system_health()
                    
                # Update status
                self._update_status("running")
                
                # Sleep for monitoring interval
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Daemon error: {e}", exc_info=True)
                time.sleep(60)  # Wait longer on error
                
    def _start_ominari_system(self):
        """Start the Ominari trading system."""
        try:
            # Use uv to run the system
            cmd = ["uv", "run", "python", "run_ominari_system.py"]
            
            self.ominari_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=Path(__file__).parent,
                env=os.environ.copy()
            )
            
            logger.info(f"Started Ominari system with PID {self.ominari_process.pid}")
            
            # Give it time to start
            time.sleep(10)
            
            # Check if it started successfully
            if self.ominari_process.poll() is None:
                logger.info("Ominari system started successfully")
            else:
                stdout, stderr = self.ominari_process.communicate(timeout=1)
                logger.error(f"Ominari system failed to start: {stderr.decode()}")
                self.ominari_process = None
                
        except Exception as e:
            logger.error(f"Failed to start Ominari system: {e}")
            self.ominari_process = None
            
    def _stop_ominari_system(self):
        """Stop the Ominari trading system."""
        if not self.ominari_process:
            return
            
        logger.info("Stopping Ominari system")
        
        # Send SIGTERM for graceful shutdown
        self.ominari_process.terminate()
        
        try:
            # Wait for graceful shutdown
            self.ominari_process.wait(timeout=30)
            logger.info("Ominari system stopped gracefully")
        except subprocess.TimeoutExpired:
            # Force kill if necessary
            logger.warning("Force killing Ominari system")
            self.ominari_process.kill()
            self.ominari_process.wait()
            
        self.ominari_process = None
        
    def _is_ominari_running(self):
        """Check if Ominari system is running."""
        if not self.ominari_process:
            return False
            
        # Check if process is still alive
        return self.ominari_process.poll() is None
        
    def _check_system_health(self):
        """Check Ominari system health."""
        try:
            # Check if the main Ominari process is running
            if self.ominari_process and self.ominari_process.poll() is None:
                # Process is still running
                return True
            else:
                logger.error("Ominari process has stopped")
                return False
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
            
    def _update_status(self, status):
        """Update daemon status file."""
        try:
            status_data = {
                "status": status,
                "pid": os.getpid(),
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "uptime_seconds": (datetime.now(timezone.utc) - self.start_time).total_seconds() if self.start_time else 0,
                "ominari_pid": self.ominari_process.pid if self.ominari_process else None,
                "last_update": datetime.now(timezone.utc).isoformat()
            }
            
            with open(STATUS_FILE, 'w') as f:
                json.dump(status_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update status: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ominari System Daemon")
    parser.add_argument('command', choices=['start', 'stop', 'status', 'restart'],
                       help='Daemon command')
    args = parser.parse_args()
    
    daemon = OminariDaemon()
    
    if args.command == 'start':
        if daemon.start():
            print("Ominari daemon started")
        else:
            print("Failed to start daemon (already running?)")
            sys.exit(1)
            
    elif args.command == 'stop':
        # Find and stop running daemon
        if PID_FILE.exists():
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                print("Ominari daemon stopped")
            except ProcessLookupError:
                print("Daemon not running")
                PID_FILE.unlink()
        else:
            print("Daemon not running")
            
    elif args.command == 'status':
        if STATUS_FILE.exists():
            with open(STATUS_FILE, 'r') as f:
                status = json.load(f)
            print(f"Status: {status['status']}")
            print(f"PID: {status.get('pid', 'N/A')}")
            print(f"Uptime: {status.get('uptime_seconds', 0):.0f} seconds")
            print(f"Ominari PID: {status.get('ominari_pid', 'N/A')}")
        else:
            print("Status: stopped")
            
    elif args.command == 'restart':
        # Stop then start
        if PID_FILE.exists():
            with open(PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)  # Wait for shutdown
            except ProcessLookupError:
                pass
                
        if daemon.start():
            print("Ominari daemon restarted")
        else:
            print("Failed to restart daemon")
            sys.exit(1)


if __name__ == "__main__":
    main()
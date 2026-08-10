# scheduler/reminder.py

"""
Scheduler — OS-level daily trigger for the capture pipeline.

Responsibilities:
- Register a task function to run once per day at a configured time
- Run a blocking loop that checks and dispatches pending jobs
- Isolate task failures so the scheduler never crashes
- Provide graceful Ctrl+C shutdown

Design contract:
- Uses the `schedule` library (pure Python, no system cron needed)
- Time is read from config.settings.SCHEDULE_TIME (HH:MM, 24h)
- The task function is injected by the caller — this module has
  no knowledge of voice, NLP, or Notion
- One module = one responsibility: timing only
"""

import logging
import time
import threading
from typing import Callable

import schedule

from config.settings import SCHEDULE_TIME

logger = logging.getLogger("chitragupta.scheduler")


# ── Time validation ───────────────────────────────────────────────────────────

def _validate_time(time_str: str) -> str:
    """
    Confirm that SCHEDULE_TIME is a valid HH:MM 24-hour string.

    Args:
        time_str: Raw value from settings, e.g. "09:00" or "21:30".

    Returns:
        The validated time string unchanged.

    Raises:
        SchedulerConfigError: if the format is wrong or values are out of range.
    """
    parts = time_str.strip().split(":")

    if len(parts) != 2:
        raise SchedulerConfigError(
            f"SCHEDULE_TIME '{time_str}' is not in HH:MM format. "
            "Example: '09:00' or '21:30'."
        )

    try:
        hh, mm = int(parts[0]), int(parts[1])
    except ValueError:
        raise SchedulerConfigError(
            f"SCHEDULE_TIME '{time_str}' contains non-integer parts. "
            "Use digits only, e.g. '09:00'."
        )

    if not (0 <= hh <= 23):
        raise SchedulerConfigError(
            f"SCHEDULE_TIME hour '{hh}' is out of range (0–23)."
        )
    if not (0 <= mm <= 59):
        raise SchedulerConfigError(
            f"SCHEDULE_TIME minute '{mm}' is out of range (0–59)."
        )

    return f"{hh:02d}:{mm:02d}"


# ── Task wrapper ──────────────────────────────────────────────────────────────

def _safe_run(task: Callable[[], None]) -> None:
    """
    Execute the task function inside a try/except so that any exception
    in the task does NOT propagate to the scheduler loop.

    Logs the full traceback for debugging without crashing the process.
    """
    logger.info("Scheduler: firing scheduled task.")
    try:
        task()
        logger.info("Scheduler: task completed successfully.")
    except KeyboardInterrupt:
        # Let Ctrl+C propagate upward to the scheduler loop
        raise
    except Exception:
        # Catch-all: log it, keep scheduler alive
        logger.exception(
            "Scheduler: task raised an unhandled exception. "
            "Scheduler remains running — next execution at %s.",
            SCHEDULE_TIME,
        )


# ── Public API ────────────────────────────────────────────────────────────────

def run_scheduler(task_function: Callable[[], None]) -> None:
    """
    Register the task to run once daily at SCHEDULE_TIME and start
    the blocking scheduler loop.

    This is the primary entry point. Call it from main.py when the user
    wants to run Chitragupta in scheduled / daemon mode.

    The loop runs on the calling thread and blocks until Ctrl+C.

    Args:
        task_function: Zero-argument callable — typically the full
                       capture pipeline (voice → NLP → JSON → Notion).

    Raises:
        SchedulerConfigError: if SCHEDULE_TIME is malformed.
    """
    validated_time = _validate_time(SCHEDULE_TIME)

    # Clear any previously registered jobs (safe for repeated calls in tests)
    schedule.clear()

    schedule.every().day.at(validated_time).do(
        _safe_run, task=task_function
    )

    next_run = schedule.next_run()
    logger.info(
        "Scheduler registered | time=%s  next_run=%s",
        validated_time, next_run,
    )

    print(f"\n  🕐  Chitragupta scheduler started.")
    print(f"      Daily trigger set for {validated_time}.")
    print(f"      Next run: {next_run}")
    print("      Press Ctrl+C to stop.\n")

    start_scheduler(task_function)


def start_scheduler(task_function: Callable[[], None]) -> None:
    """
    Start the blocking scheduler loop.

    Separated from run_scheduler() so callers can register custom
    schedule patterns (e.g. every().hour) before entering the loop.

    Args:
        task_function: Passed through only for the shutdown log message.
                       Jobs must already be registered before calling this.
    """
    logger.info("Scheduler loop starting.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(30)   # check every 30 s — low CPU, acceptable latency

    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user (KeyboardInterrupt).")
        print("\n  ✋  Scheduler stopped. Goodbye.\n")

    except Exception:
        logger.exception("Scheduler loop encountered an unexpected error.")
        raise


def run_now(task_function: Callable[[], None]) -> None:
    """
    Execute the task immediately, outside the scheduled loop.

    Useful for:
    - Manual / on-demand capture from main.py
    - Testing the pipeline without waiting for the scheduled time

    Args:
        task_function: The capture pipeline callable.
    """
    logger.info("run_now: executing task immediately.")
    print("\n  ▶  Running capture pipeline now ...\n")
    _safe_run(task_function)


def run_in_background(task_function: Callable[[], None]) -> threading.Thread:
    """
    Start the scheduler loop on a daemon thread so the main thread
    stays free for interactive use (e.g. a CLI menu).

    The daemon thread stops automatically when the main process exits.

    Args:
        task_function: Zero-argument callable to register and run daily.

    Returns:
        The running daemon Thread (caller can join() if needed).

    Raises:
        SchedulerConfigError: if SCHEDULE_TIME is malformed.
    """
    validated_time = _validate_time(SCHEDULE_TIME)
    schedule.clear()

    schedule.every().day.at(validated_time).do(
        _safe_run, task=task_function
    )

    logger.info(
        "Background scheduler registered | time=%s", validated_time
    )

    thread = threading.Thread(
        target=start_scheduler,
        args=(task_function,),
        daemon=True,
        name="chitragupta-scheduler",
    )
    thread.start()

    logger.info("Scheduler running in background thread.")
    print(f"\n  🕐  Background scheduler active — daily trigger at {validated_time}.\n")

    return thread


# ── Custom exceptions ─────────────────────────────────────────────────────────

class SchedulerConfigError(Exception):
    """Raised when SCHEDULE_TIME is malformed or out of range."""
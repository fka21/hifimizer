# utils/subprocess_logger.py
import os
import signal
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple

# Conventional exit code used by GNU `timeout` to signal "the command timed out".
TIMEOUT_EXIT_CODE = 124

#: Every process this module has started and not yet reaped. Because each is
#: launched with ``start_new_session=True`` it lives in its own process group,
#: which a naive ``psutil.Process().children()`` sweep from the parent can miss.
#: The signal handler in hifimizer drains this so Ctrl-C actually stops hifiasm.
_ACTIVE_PROCESSES = set()
_ACTIVE_LOCK = threading.Lock()


class SubprocessLogger:
    """
    Utility class to run subprocesses with dedicated logging.

    The process is started in its own session (``start_new_session=True``) so
    that on timeout the *entire* process tree can be signalled with
    ``killpg``.  This matters for tools like BUSCO, which spawn metaeuk /
    miniprot / augustus / hmmsearch children: signalling only the direct child
    leaves those orphans running and holding CPUs.
    """

    def __init__(self, logs_dir: Path = None):
        if logs_dir is None:
            logs_dir = Path.cwd() / "logs"
        logs_dir = Path(logs_dir)
        self.logs_dir = logs_dir if logs_dir.is_absolute() else logs_dir.resolve()
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ kill
    @classmethod
    def kill_all_active(cls) -> int:
        """
        Kill every subprocess this module started that is still running.

        Called from hifimizer's SIGINT/SIGTERM handler. Returns how many
        processes were signalled.
        """
        with _ACTIVE_LOCK:
            processes = list(_ACTIVE_PROCESSES)

        killed = 0
        for process in processes:
            if process.poll() is None:
                cls._kill_process_tree(process)
                killed += 1

        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.difference_update(processes)
        return killed

    @staticmethod
    def _kill_process_tree(process) -> None:
        """SIGKILL the process group, then sweep any survivors with psutil."""
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            pass

        # Belt and braces: anything that escaped the process group (e.g. a
        # child that called setsid itself) is caught here.
        try:
            import psutil

            parent = psutil.Process(process.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except Exception:
                    pass
            try:
                parent.kill()
            except Exception:
                pass
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

        # Reap, so we do not leave a zombie behind.
        try:
            process.wait(timeout=30)
        except Exception:
            pass

    # ------------------------------------------------------------------- run
    def run_command_with_logging(
        self,
        command: str,
        log_filename: str,
        command_name: str = "subprocess",
        trial_id: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        cwd: Optional[Path] = None,
    ) -> Tuple[int, str]:
        """
        Run a command with output redirected to a log file.

        Args:
            command: Shell command to execute
            log_filename: Name of the log file (without path)
            command_name: Human-readable name for the command
            trial_id: Optional trial ID for naming
            timeout_seconds: Wall-clock limit. On expiry the whole process
                group is killed and ``TIMEOUT_EXIT_CODE`` (124) is returned.
            cwd: Working directory for the command

        Returns:
            Tuple of (return_code, log_file_path)
        """
        if trial_id is not None:
            log_filename = f"trial_{trial_id}_{log_filename}"

        log_file_path = self.logs_dir / log_filename

        with open(log_file_path, "a") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Command: {command_name}\n")
            f.write(f"Full command: {command}\n")
            if cwd:
                f.write(f"Working directory: {cwd}\n")
            if timeout_seconds:
                f.write(f"Timeout: {timeout_seconds:.0f} s\n")
            f.write(f"Timestamp: {self._get_timestamp()}\n")
            f.write(f"{'=' * 60}\n\n")

        try:
            with open(log_file_path, "a") as f:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    executable="/bin/bash",  # commands use `set -o pipefail`
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(cwd) if cwd else None,
                    start_new_session=True,  # own process group -> killpg works
                )
                with _ACTIVE_LOCK:
                    _ACTIVE_PROCESSES.add(process)
                try:
                    return_code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    self._kill_process_tree(process)
                    with open(log_file_path, "a") as lf:
                        lf.write(
                            f"\nERROR: Command '{command_name}' timed out after "
                            f"{timeout_seconds} seconds; process group killed.\n"
                        )
                    return TIMEOUT_EXIT_CODE, str(log_file_path)
                finally:
                    with _ACTIVE_LOCK:
                        _ACTIVE_PROCESSES.discard(process)

            return return_code, str(log_file_path)

        except Exception as e:
            with open(log_file_path, "a") as f:
                f.write(f"\nERROR: {str(e)}\n")
            raise RuntimeError(f"Command '{command_name}' failed: {e}")

    @staticmethod
    def _get_timestamp() -> str:
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
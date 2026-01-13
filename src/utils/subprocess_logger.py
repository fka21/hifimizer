# utils/subprocess_logger.py
import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple


class SubprocessLogger:
    """
    Utility class to run subprocesses with dedicated logging.
    """

    def __init__(self, logs_dir: Path = None):
        if logs_dir is None:
            # When called without logs_dir, don't create any default directory
            # This prevents root-level logs creation. Caller must pass logs_dir.
            logs_dir = Path.cwd() / "logs"
        self.logs_dir = (
            logs_dir.resolve()
            if isinstance(logs_dir, Path) and not logs_dir.is_absolute()
            else logs_dir
        )
        if isinstance(self.logs_dir, Path):
            self.logs_dir.mkdir(parents=True, exist_ok=True)

    def run_command_with_logging(
        self,
        command: str,
        log_filename: str,
        command_name: str = "subprocess",
        trial_id: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> Tuple[int, str]:
        """
        Run a command with output redirected to a log file.

        Args:
            command: Shell command to execute
            log_filename: Name of the log file (without path)
            command_name: Human-readable name for the command
            trial_id: Optional trial ID for naming

        Returns:
            Tuple of (return_code, log_file_path)
        """
        # Create trial-specific log filename if trial_id provided
        if trial_id is not None:
            log_filename = f"trial_{trial_id}_{log_filename}"

        log_file_path = self.logs_dir / log_filename

        # Write command header to log file
        with open(log_file_path, "a") as f:
            f.write(f"\n{'=' * 60}\n")
            f.write(f"Command: {command_name}\n")
            f.write(f"Full command: {command}\n")
            f.write(f"Timestamp: {self._get_timestamp()}\n")
            f.write(f"{'=' * 60}\n\n")

        # Run command with output redirection
        try:
            with open(log_file_path, "a") as f:
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                try:
                    return_code = process.wait(timeout=timeout_seconds)
                except subprocess.TimeoutExpired:
                    # Attempt to kill process tree if psutil is available
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
                        # Fallback: kill the process
                        try:
                            process.kill()
                        except Exception:
                            pass

                    with open(log_file_path, "a") as lf:
                        lf.write(
                            f"\nERROR: Command timed out after {timeout_seconds} seconds.\n"
                        )

                    # Use conventional timeout exit code 124 to indicate timeout
                    return 124, str(log_file_path)

            return return_code, str(log_file_path)

        except Exception as e:
            with open(log_file_path, "a") as f:
                f.write(f"\nERROR: {str(e)}\n")
            raise RuntimeError(f"Command '{command_name}' failed: {e}")

    @staticmethod
    def _get_timestamp() -> str:
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

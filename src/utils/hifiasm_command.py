import logging
import subprocess
from utils.subprocess_logger import SubprocessLogger
from pathlib import Path

logger = logging.getLogger(__name__)


def build_hifiasm_command(
    prefix="trial_assembly",
    x=None,
    y=None,
    s=None,
    n=None,
    m=None,
    p=None,
    haploid_genome_size=None,
    threads=None,
    sensitive=False,
    D=None,
    N=None,
    max_kocc=None,
    hic1=None,
    hic2=None,
    ul=None,
    s_base=None,
    f_perturb=None,
    l_msjoin=None,
    path_max=None,
    path_min=None,
    primary=False,
    default_only=False,
    **extra_args,
):
    """
    Constructs the hifiasm command string based on given parameters.

    Args:
        prefix (str): Output prefix.
        x, y, s (float): Hifiasm parameters controlling graph simplification.
        n, m, p (int): Hifiasm internal parameters.
        haploid_genome_size (int): Haploid genome size in megabases.
        threads (int): Number of threads to use.
        sensitive (bool): Whether to enable sensitivity parameters.
        D, N, max_kocc (int): Sensitivity-specific tuning parameters.
        default_only (bool): If True, build only the default minimal command.

    Returns:
        str: Command-line string for running hifiasm.
    """
    if None in [haploid_genome_size, threads]:
        raise ValueError("haploid_genome_size and threads must be provided.")

    cmd = f"hifiasm -o {prefix} --hg-size {haploid_genome_size}m -t {threads} "

    if primary:
        cmd += "--primary "

    if default_only:
        return cmd.strip()

    if not None in [x, y, s, n, m, p]:
        cmd += f"-x {x} -y {y} -s {s} -n {n} -m {m} -p {p} "

    if hic1 and hic2:
        cmd += f"--h1 {hic1} --h2 {hic2} "
        if s_base is not None:
            cmd += f"--s-base {s_base} "
        if f_perturb is not None:
            cmd += f"--f-perturb {f_perturb} "
        if l_msjoin is not None:
            cmd += f"--l-msjoin {l_msjoin} "

    if ul:
        cmd += f"--ul {ul} "
        if path_max is not None:
            cmd += f"--path-max {path_max} "
        if path_min is not None:
            cmd += f"--path-min {path_min} "

    if sensitive:
        if not None in [D, N, max_kocc]:
            cmd += f"-D {D} -N {N} --max-kocc {max_kocc} "

    return cmd.strip()


def run_default_hifiasm_assembly(
    prefix,
    haploid_genome_size,
    threads,
    primary=False,
    hic1=None,
    hic2=None,
    ul=None,
    input_reads=None,
):
    """
    Run a clean hifiasm assembly with default parameters only.
    """
    try:
        command = (
            build_hifiasm_command(
                prefix=prefix,
                haploid_genome_size=haploid_genome_size,
                threads=threads,
                primary=primary,
                hic1=hic1,
                hic2=hic2,
                ul=ul,
                default_only=True,
            )
            + f" {input_reads}"
        )
    except ValueError as e:
        logger.error(f"Failed to build default command: {e}")
        exit(1)

    logger.info(f"Running clean hifiasm assembly with parameters:\n{command}")

    try:
        return_code, log_path = loggerClass.run_command_with_logging(
            command=command, log_filename="hifiasm.log", command_name="hifiasm"
        )
    except RuntimeError as e:
        logger.error(str(e))
        exit(1)

    if return_code == 0:
        logger.info("Clean hifiasm run completed successfully")
    else:
        logger.error(
            f"Default run failed with code: {return_code}. Check log at {log_path}"
        )
        exit(1)

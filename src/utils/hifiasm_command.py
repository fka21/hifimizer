import logging
import shutil
from pathlib import Path

from utils.subprocess_logger import SubprocessLogger

logger = logging.getLogger(__name__)

#: Every tunable hifiasm parameter hifimizer knows how to replay from a
#: recorded trial. Kept here, next to ``build_hifiasm_command``, so that adding
#: a parameter is a single-file change.
HIFIASM_PARAM_KEYS = (
    "x", "y", "s", "n", "m", "p", "u",
    "D", "N", "max_kocc",
    "s_base", "f_perturb", "l_msjoin",
    "path_max", "path_min",
)


def collect_hifiasm_outputs(prefix, dest_dir, label) -> int:
    """
    Copy the hifiasm outputs sitting at ``prefix`` into ``dest_dir``.

    Every trial shares one hifiasm ``-o`` prefix so the ``*.bin`` files are
    reused, which means a result has to be copied out before the next trial
    overwrites it. The ``.bin`` files themselves are deliberately not copied:
    they are large, and they are the thing we want left in place.

    Returns:
        Number of files copied.
    """
    prefix = Path(prefix)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for f in sorted(prefix.parent.glob(f"{prefix.name}*")):
        if not f.is_file() or f.suffix == ".bin":
            continue
        target = dest_dir / f"{label}{f.name[len(prefix.name):]}"
        try:
            shutil.copy2(f, target)
            copied += 1
        except Exception as e:
            logger.warning(f"Could not copy {f.name} -> {target}: {e}")

    logger.info(f"Copied {copied} file(s) to {dest_dir}")
    return copied


def build_hifiasm_command(
    prefix="trial_assembly",
    x=None,
    y=None,
    s=None,
    n=None,
    m=None,
    p=None,
    u=None,
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
    ont=False,
    hom_cov=None,
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

    if hom_cov is not None:
        cmd += f"--hom-cov {hom_cov} "

    if ont:
        cmd += "--ont "

    if hic1 and hic2:
        cmd += f"--h1 {hic1} --h2 {hic2} "

    if ul:
        cmd += f"--ul {ul} "

    if default_only:
        return cmd.strip()

    # --- tunable parameters ----------------------------------------------
    # All six are emitted together or not at all: hifiasm's graph-cleaning
    # parameters are only meaningful as a set.
    if all(v is not None for v in (x, y, s, n, m, p)):
        cmd += f"-x {x} -y {y} -s {s} -n {n} -m {m} -p {p} "

    if u is not None:
        cmd += f"-u {u} "

    if hic1 and hic2:
        if s_base is not None:
            cmd += f"--s-base {s_base} "
        if f_perturb is not None:
            cmd += f"--f-perturb {f_perturb} "
        if l_msjoin is not None:
            cmd += f"--l-msjoin {l_msjoin} "

    if ul:
        if path_max is not None:
            cmd += f"--path-max {path_max} "
        if path_min is not None:
            cmd += f"--path-min {path_min} "

    if sensitive:
        if all(v is not None for v in (D, N, max_kocc)):
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
    ont=False,
    logs_dir=None,
    hom_cov=None,
    walltime_hours=None,
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
                ont=ont,
                hom_cov=hom_cov,
            )
            + f" {input_reads}"
        )
    except ValueError as e:
        logger.error(f"Failed to build default command: {e}")
        exit(1)

    logger.info(f"Running clean hifiasm assembly with parameters:\n{command}")

    try:
        subprocess_logger = SubprocessLogger(
            logs_dir=logs_dir if logs_dir else Path(prefix).parent / "logs"
        )
        return_code, log_path = subprocess_logger.run_command_with_logging(
            command=command,
            log_filename="hifiasm.log",
            command_name="hifiasm",
            timeout_seconds=walltime_hours * 3600 if walltime_hours else None,
        )
    except RuntimeError as e:
        logger.error(str(e))
        exit(1)

    if return_code == 124:
        logger.error(
            f"Default hifiasm run exceeded the walltime limit ({walltime_hours} h) "
            f"and was killed. Check log at {log_path}"
        )
        exit(1)

    if return_code == 0:
        logger.info("Clean hifiasm run completed successfully")
    else:
        logger.error(
            f"Default run failed with code: {return_code}. Check log at {log_path}"
        )
        exit(1)
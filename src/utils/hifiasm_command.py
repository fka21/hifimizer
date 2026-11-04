import logging
import shlex
import shutil
from pathlib import Path

from utils.subprocess_logger import SubprocessLogger

logger = logging.getLogger(__name__)

#: Every tunable hifiasm parameter hifimizer knows how to replay from a
#: recorded trial. Kept here, next to ``build_hifiasm_command``, so that adding
#: a parameter is a single-file change.
HIFIASM_PARAM_KEYS = (
    "x", "y", "s", "n", "m", "p", "u",
    "D", "N", "max_kocc", "k", "w", "max_od_ec", "max_od_final",
    "s_base", "f_perturb", "l_msjoin",
    "path_max", "path_min",
    "dual_scaf",
)

#: hifiasm flag -> (hifimizer parameter name, value type). ``None`` as the type
#: marks a switch that takes no value. Used to recognise when a user-supplied
#: argument is pinning something hifimizer would otherwise be optimizing.
HIFIASM_FLAG_TO_PARAM = {
    "-x": ("x", float),
    "-y": ("y", float),
    "-s": ("s", float),
    "-n": ("n", int),
    "-m": ("m", int),
    "-p": ("p", int),
    "-u": ("u", int),
    "-D": ("D", int),
    "-N": ("N", int),
    "--max-kocc": ("max_kocc", int),
    "-k": ("k", int),
    "-w": ("w", int),
    "--max-od-ec": ("max_od_ec", float),
    "--max-od-final": ("max_od_final", float),
    "--s-base": ("s_base", float),
    "--f-perturb": ("f_perturb", float),
    "--l-msjoin": ("l_msjoin", int),
    "--path-max": ("path_max", float),
    "--path-min": ("path_min", float),
    "--dual-scaf": ("dual_scaf", None),
}

#: Flags hifimizer sets itself, from dedicated command-line options. Accepting
#: them here would mean two sources of truth for the same value and a command
#: with the flag twice, so they are refused with a pointer to the real option.
RESERVED_HIFIASM_FLAGS = {
    "-o": "--output-dir",
    "-t": "--threads",
    "--hg-size": "--genome-size",
    "--h1": "--hic1",
    "--h2": "--hic2",
    "--ul": "--ul",
    "--primary": "--primary",
    "--ont": "--ont",
    "--hom-cov": "--hom-cov",
}


def parse_hifiasm_extra(extra):
    """
    Split user-supplied hifiasm arguments into pinned parameters and passthrough.

    Anything hifimizer knows how to optimize is returned as a parameter value
    and is then **fixed for every run**: it is dropped from the search space
    rather than sampled, so the user's value wins. Anything else is returned as
    raw tokens and appended to every hifiasm command unchanged, which is how a
    parameter hifimizer has never heard of still reaches the assembler.

    Args:
        extra: A raw argument string, e.g. ``"-s 0.5 --telo-m CCCTAA"``.

    Returns:
        ``(pinned, passthrough)`` -- a dict of parameter name to typed value,
        and a list of argument tokens.

    Raises:
        ValueError: for a flag hifimizer sets itself, a known flag with a
            missing or unparseable value, or a value where a switch was
            expected.
    """
    if not extra:
        return {}, []

    tokens = shlex.split(extra) if isinstance(extra, str) else [str(t) for t in extra]
    pinned, passthrough = {}, []

    index = 0
    while index < len(tokens):
        token = tokens[index]

        # Accept --flag=value as well as --flag value.
        flag, inline_value = (
            token.split("=", 1) if token.startswith("-") and "=" in token
            else (token, None)
        )

        if flag in RESERVED_HIFIASM_FLAGS:
            raise ValueError(
                f"hifiasm's {flag} is set by hifimizer itself; use "
                f"{RESERVED_HIFIASM_FLAGS[flag]} instead of passing it here."
            )

        if flag in HIFIASM_FLAG_TO_PARAM:
            name, kind = HIFIASM_FLAG_TO_PARAM[flag]
            if kind is None:
                if inline_value is not None:
                    raise ValueError(f"hifiasm's {flag} is a switch and takes no value.")
                pinned[name] = True
                index += 1
                continue

            if inline_value is not None:
                raw = inline_value
                index += 1
            else:
                if index + 1 >= len(tokens):
                    raise ValueError(f"hifiasm's {flag} needs a value.")
                raw = tokens[index + 1]
                index += 2
            try:
                pinned[name] = kind(raw)
            except ValueError:
                raise ValueError(
                    f"Could not read '{raw}' as a value for {flag} "
                    f"(expected {kind.__name__})."
                ) from None
            continue

        passthrough.append(token)
        index += 1

    return pinned, passthrough


def dual_scaf_supported(hic1=None, hic2=None, ul=None, primary=False):
    """
    Whether ``--dual-scaf`` can change the assembly hifimizer scores.

    hifiasm only scaffolds inside its dual-haplotype output path
    (``output_trio_graph_joint`` in 0.25.0's Overlaps.cpp), so it can only
    matter where hap1/hap2 are what gets scored: Hi-C and ultra-long runs.
    In a plain HiFi run the scored ``bp.p_ctg`` is written before that step,
    and ``--primary`` produces no haplotype pair at all, so in both it would
    be a search dimension that cannot move the score.
    """
    if primary:
        return False
    return bool((hic1 and hic2) or ul)


def delete_hifiasm_bins(prefix):
    """
    Remove every ``.bin`` file hifiasm has cached at ``prefix``.

    hifiasm loads ``*.ec.bin`` / ``*.ovlp.*.bin`` whenever they exist and then
    skips k-mer filtering, overlap and error correction entirely
    (``ha_assemble_pair`` in 0.25.0's Assembly.cpp). Those are exactly the
    stages ``-D``, ``-N`` and ``--max-kocc`` act on, so in ``--sensitive``
    mode the cache has to go before every run or the sampled values are
    silently ignored. The Hi-C and ultra-long caches go too: both are built
    on the corrected reads, so they would be stale the moment those change.

    Returns:
        The names of the files removed.
    """
    prefix = Path(prefix)
    removed = []
    for f in sorted(prefix.parent.glob(f"{prefix.name}*.bin")):
        try:
            f.unlink()
            removed.append(f.name)
        except FileNotFoundError:
            pass
    return removed


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
    k=None,
    w=None,
    max_od_ec=None,
    max_od_final=None,
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
    dual_scaf=False,
    extra_args=None,
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
        k, w (int), max_od_ec, max_od_final (float): Sensitivity-specific
            overlap and error-correction parameters (-k, -w, --max-od-ec,
            --max-od-final); each is emitted only when given. hifiasm
            refuses ``max_od_ec < max_od_final``, so ``max_od_final`` is
            capped at ``max_od_ec`` when both are given.
        default_only (bool): If True, build only the default minimal command.
        dual_scaf (bool): Emit --dual-scaf, which lets hifiasm use the
            homologous region in one haplotype to scaffold across a gap in the
            other. Only emitted where it can change the scored assembly -- Hi-C
            and ultra-long runs, never with ``--primary``; see
            :func:`dual_scaf_supported`.
        extra_args (list): User-supplied hifiasm arguments, appended verbatim
            to every command including the default-parameter baseline. They
            describe the experiment rather than the search, so leaving them off
            trial 0 would make it a baseline for a different assembly.

    Returns:
        str: Command-line string for running hifiasm.
    """
    if None in [haploid_genome_size, threads]:
        raise ValueError("haploid_genome_size and threads must be provided.")

    # Passed in kilobases: --hg-size takes an integer with a k/m/g suffix, and
    # whole megabases would round small genomes (4.64 Mb -> 5m).
    hg_size_kb = max(1, round(float(haploid_genome_size) * 1000))
    cmd = f"hifiasm -o {prefix} --hg-size {hg_size_kb}k -t {threads} "

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

    # User arguments go last so that if one ever collides with something built
    # above, hifiasm's getopt takes the later occurrence -- the user's.
    tail = (" " + " ".join(str(a) for a in extra_args)) if extra_args else ""

    # Emitted before the default_only return: a user-pinned argument applies to
    # the baseline too, otherwise trial 0 is a baseline for a different assembly.
    if dual_scaf and dual_scaf_supported(hic1, hic2, ul, primary):
        cmd += "--dual-scaf "

    if default_only:
        return (cmd.strip() + tail).strip()

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
        # hifiasm exits with "max_ov_diff_ec shouldn't be smaller than
        # max_ov_diff_final" (CommandLines.cpp) otherwise. Capping here, where
        # every trial, rerun and final rebuild builds its command, keeps all of
        # them identical for the same recorded parameters.
        if max_od_ec is not None and max_od_final is not None:
            if max_od_final > max_od_ec:
                logger.info(
                    f"--max-od-final {max_od_final} capped at --max-od-ec "
                    f"{max_od_ec}: hifiasm requires max-od-ec >= max-od-final."
                )
                max_od_final = max_od_ec
        for flag, value in (
            ("-k", k),
            ("-w", w),
            ("--max-od-ec", max_od_ec),
            ("--max-od-final", max_od_final),
        ):
            if value is not None:
                cmd += f"{flag} {value} "

    return (cmd.strip() + tail).strip()


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
    dual_scaf=False,
    extra_args=None,
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
                dual_scaf=dual_scaf,
                extra_args=extra_args,
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
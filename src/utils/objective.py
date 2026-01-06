import subprocess
import shutil
import logging
from pathlib import Path
from utils.hifiasm_command import build_hifiasm_command
from utils.subprocess_logger import SubprocessLogger
from utils.assembly_eval import AssemblyEvaluator

class ObjectiveBuilder:
    def __init__(self, evaluator, input_reads, haploid_genome_size, threads, 
                 hic1=None, hic2=None, ul=None, sensitive=False, primary=False, include_busco=True,
                 busco_lineage="metazoa_odb12", download_path=None, output_dir=None):
        """
        Initialize the objective builder with necessary configuration.
        
        Args:
            evaluator: Assembly evaluator instance
            input_reads: Path to input reads
            haploid_genome_size: Genome size
            threads: Number of threads to use
            hic1: Path to Hi-C R1 reads file (optional)
            hic2: Path to Hi-C R2 reads file (optional)
            ul: Path to ultra-long ONT reads file (optional)
            sensitive: Whether to optimize for sensitivity
            primary: Whether to perform primary assembly only
            include_busco: Whether to include BUSCO metrics
        """
        self.evaluator = evaluator
        self.input_reads = input_reads
        self.haploid_genome_size = haploid_genome_size
        self.threads = threads
        self.hic1 = hic1
        self.hic2 = hic2
        self.ul = ul
        self.sensitive = sensitive
        self.primary = primary
        self.include_busco = include_busco
        self.busco_lineage = busco_lineage
        self.download_path = download_path
        self.subprocess_logger = SubprocessLogger()
        
        # Set up output directory for default assembly results
        if output_dir is None:
            self.output_dir = Path.cwd()
        else:
            self.output_dir = Path(output_dir).resolve()
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def build_objective(self):
        """
        Build and return the objective function for Optuna optimization.
        """
        def objective(trial):
            # Get trial number for logging
            trial_id = trial.number

            # Initialize evaluator with trial ID
            evaluator = AssemblyEvaluator(
                known_genome_size=self.evaluator.known_genome_size,
                input_reads=self.evaluator.input_reads,
                threads=self.threads,
                trial_id=trial_id,
                download_path=self.download_path
            ) 
            
            # Define base assembly name
            base_name = "trial_assembly"

            # Choose the suffix based on inputs
            if self.hic1 and self.hic2:
                suffix = "hic.hap1.p_ctg"
            elif self.ul:
                suffix = "bp.hap1.p_ctg"
            else:
                suffix = "bp.p_ctg"  

            # Define file names
            gfa_file = f"{base_name}.{suffix}.gfa"
            fasta_file = f"{base_name}.{suffix}.fasta"
            
            # Assenbly to compare to should be the assembly generated with default values
            if trial_id == 0:  
                
                params = {
                    'haploid_genome_size': self.haploid_genome_size,
                    'threads': self.threads,
                    'sensitive': self.sensitive,
                    'hic1': self.hic1,
                    'hic2': self.hic2,
                    'ul': self.ul,
                    'primary': self.primary
                }
                # Only keep non-None params
                params = {k: v for k, v in params.items() if v is not None}

                command = build_hifiasm_command(**params)
                command += f" {self.input_reads}"
            else:
                
                # Parameters to optimize by default
                x = trial.suggest_float('x', 0.59, 0.99, step=0.01)
                y = trial.suggest_float('y', 0.01, 0.41, step=0.01)
                s = trial.suggest_float('s', 0.55, 1, step=0.01)
                n = trial.suggest_int('n', 0, 10)
                m = trial.suggest_int('m', 500_000, 20_000_000, step = 10_000)  
                p = trial.suggest_int('p', 0, 10_000, step=200)
                
                hic_params = {}
                ont_params = {}
                
                # Hi-C parameter space
                if self.hic1 and self.hic2:
                    hic_params.update({
                        's_base': trial.suggest_float('s_base', 0, 1, step=0.05),
                        'f_perturb': trial.suggest_float('f_perturb', 0, 1, step=0.05),
                        'l_msjoin': trial.suggest_int('l_msjoin', 0, 10_000_000, step=100_000)
                    })
                
                # UL ONT parameter space
                if self.ul:
                    ont_params.update({
                        'path_max': trial.suggest_float('path_max', 0.0, 1.0, step=0.05),
                        'path_min': trial.suggest_float('path_min', 0.0, 1.0, step=0.05)
                    })
                
                if self.sensitive:
                    D = trial.suggest_int('D', 3, 20, step=1)
                    N = trial.suggest_int('N', 50, 400, step=10)
                    max_kocc = trial.suggest_int('max_kocc', 1000, 5000, step=100)
                    
                    command = build_hifiasm_command(
                        x=x, y=y, s=s, n=n, m=m, p=p,
                        haploid_genome_size=self.haploid_genome_size,
                        threads=self.threads,
                        sensitive=True,
                        D=D, N=N, max_kocc=max_kocc,
                        hic1=self.hic1, hic2=self.hic2, ul=self.ul,
                        **hic_params,
                        **ont_params,
                        primary=self.primary
                    )
                else:
                    command = build_hifiasm_command(
                        x=x, y=y, s=s, n=n, m=m, p=p,
                        haploid_genome_size=self.haploid_genome_size,
                        threads=self.threads,
                        sensitive=False,
                        hic1=self.hic1, hic2=self.hic2, ul=self.ul,
                        **hic_params,
                        **ont_params,
                        primary=self.primary
                    )

                command += f" {self.input_reads}"
            
            
            try:
                # Run hifiasm with dedicated logging
                return_code, log_path = self.subprocess_logger.run_command_with_logging(
                    command=command,
                    log_filename="hifiasm.log",
                    command_name="hifiasm",
                    trial_id=trial_id,
                    timeout_seconds=24*3600
                )
                
                if return_code != 0:
                    raise RuntimeError(f"Hifiasm failed - see {log_path}")
                
                # Check if output exists
                if not Path(gfa_file).exists():
                    raise FileNotFoundError(f"GFA file not found: {gfa_file}")
                
                # Evaluate assembly
                logging.info(f"Trial {trial_id}: Evaluating assembly")
                weighted_sum = evaluator.evaluate_assembly(
                    gfa_file=gfa_file, 
                    fasta_file=fasta_file, 
                    include_busco=self.include_busco,
                    busco_lineage = self.busco_lineage,
                    download_path=self.download_path
                )
                
                # Check if evaluation returned a valid score
                if weighted_sum is None:
                    raise RuntimeError("Evaluation returned None score")
                
                # Save results of default assembly into separate folder
                if trial_id == 0:
                    default_dir = self.output_dir / "default_assembly"
                    default_dir.mkdir(parents=True, exist_ok=True)
                    for file in Path(".").glob("trial_assembly*"):
                        shutil.copy(str(file), default_dir / file.name)
                    logging.info(f"Default setting based assembly results moved to {default_dir.resolve()}")
                
                logging.info(f"Trial {trial_id}: Completed successfully (score: {weighted_sum:.4f})")
                return weighted_sum
                
            except (TimeoutError, FileNotFoundError, RuntimeError, subprocess.SubprocessError, ValueError) as e:
                stage = self._determine_failure_stage(e, gfa_file)
                logging.error(f"Trial {trial_id}: Failed at {stage} - {str(e)}")
                import optuna
                raise optuna.exceptions.TrialPruned(f"Trial pruned at {stage}: {e}")
        
        return objective

    def _determine_failure_stage(self, error, gfa_file):
        """Determine which stage the trial failed at."""
        if "hifiasm" in str(error).lower():
            return "hifiasm assembly"
        elif not Path(gfa_file).exists():
            return "assembly output generation"
        else:
            return "assembly evaluation"
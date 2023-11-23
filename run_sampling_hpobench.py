import os
import numpy as np
import pandas as pd
import shutil
from smac import BlackBoxFacade
from omegaconf import DictConfig
from utils.logging_utils import get_logger
from utils.smac_utils import run_smac_optimization
from utils.hpobench_utils import get_run_config, get_benchmark_dict, get_task_dict
from smac.acquisition.function import LCB, EI, PI
import hydra


@hydra.main(config_path="./hydra_config", config_name="base", version_base="1.1")
def main(cfg: DictConfig):
    if "multirun" in os.getcwd():
        os.chdir("/scratch/hpc-prf-intexml/fehring/generate-smac-runs/symbolic-explanations-data-generation")

    approach = cfg["data-generation"]
    run_type = approach["run_type"]
    max_hp_comb = approach["max_hp_comb"]

    n_samples_spacing = approach["n_samples_spacing"]
    init_design_max_ratio = approach["init_design_max_ratio"]
    init_design_n_configs_per_hyperparamter = approach["init_design_n_configs_per_hyperparamter"]
    n_optimized_params = approach["n_optimized_params"]
    n_samples = approach["n_samples"]

    n_seeds = approach["n_seeds"]
    job_id = approach["job_id"]

    sampling_dir_name = "runs_sampling_hpobench"

    assert run_type == "smac", "Only SMAC sampling is supported here. All other sampling methods are not maintained in this fork"

    # number of HP combinations to consider per mode

    run_conf = get_run_config(job_id=job_id, n_optimized_params=n_optimized_params, max_hp_comb=max_hp_comb)

    task_dict = get_task_dict()
    data_set_postfix = f"_{task_dict[run_conf['task_id']]}"
    optimized_parameters = list(run_conf["hp_conf"])
    model_name = get_benchmark_dict()[run_conf["benchmark"]]
    b = run_conf["benchmark"](task_id=run_conf["task_id"], hyperparameters=optimized_parameters)

    def optimization_function_wrapper(cfg, seed):
        """Helper-function: simple wrapper to use the benchmark with smac"""
        result_dict = b.objective_function(cfg, rng=seed)
        return result_dict["function_value"]

    if run_type == "smac":
        # init design max ratio beschränkt die anzahl an hyperparmeter kombinationen die in der init design phase
        n_samples_to_eval = [
            n for n in n_samples_spacing if init_design_max_ratio * n < n_optimized_params * init_design_n_configs_per_hyperparamter
        ]  #
        if max(n_samples_spacing) not in n_samples_to_eval:
            n_samples_to_eval.append(max(n_samples_spacing))
    else:
        raise ValueError(f"Unknown or disabled run type {run_type}.")

    run_name = f"{model_name.replace(' ', '_')}_{'_'.join(optimized_parameters)}{data_set_postfix}"

    sampling_dir = f"results/{sampling_dir_name}/{run_type}"
    sampling_run_dir = f"{sampling_dir}/{run_name}"
    if os.path.exists(sampling_run_dir):
        shutil.rmtree(sampling_run_dir)
    os.makedirs(sampling_run_dir)
    if run_type == "smac":
        os.makedirs(f"{sampling_run_dir}/surrogates")

    # setup logging
    logger = get_logger(filename=f"{sampling_run_dir}/sampling_log.log")

    logger.info(f"Start {run_type} sampling for {run_name}.")

    for acquisition_function in [LCB(), EI(), PI()]:
        logger.info(f"Run: {run_name}")
        logger.info(f"Start run to sample {n_samples} samples.")

        df_samples = pd.DataFrame()

        for i in range(n_seeds):
            seed = i * 3

            np.random.seed(seed)

            # add only parameters to be optimized to configspace
            cs = b.get_configuration_space(seed=seed, hyperparameters=optimized_parameters)

            logger.info(f"Sample configs and train {model_name} with seed {seed}.")

            if run_type == "smac":
                configurations, performances, _ = run_smac_optimization(
                    configspace=cs,
                    facade=BlackBoxFacade,  # Stndard gaussian process no multi fidelity
                    acquisition_function=acquisition_function,
                    target_function=optimization_function_wrapper,
                    function_name=model_name,
                    n_eval=n_samples,
                    run_dir=sampling_run_dir,
                    seed=seed,
                    n_configs_per_hyperparamter=init_design_n_configs_per_hyperparamter,
                    max_ratio=init_design_max_ratio,
                )
            else:
                raise ValueError(f"Unknown or disabled run type {run_type}.")

            df = pd.DataFrame(data=np.concatenate((configurations.T, performances.reshape(-1, 1)), axis=1), columns=optimized_parameters + ["cost"])
            df.insert(0, "seed", seed)
            df = df.reset_index()
            df = df.rename(columns={"index": "n_samples"})
            df["n_samples"] = df["n_samples"] + 1
            df_samples = pd.concat((df_samples, df))

            df_samples.to_csv(f"{sampling_run_dir}/samples_{n_samples}.csv", index=False)


if __name__ == "__main__":
    main()

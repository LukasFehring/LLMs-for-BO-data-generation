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
from ConfigSpace import ConfigurationSpace
from smac import Callback
from smac.acquisition.function import AbstractAcquisitionFunction
from copy import deepcopy
import logging

RANDOM_CONFIGS = 100


class CustomCallback(Callback):
    def __init__(self, n_configs: int, path: str) -> None:
        super().__init__()
        self.counter = 0
        self.n_configs = n_configs
        self.path = path
        self.logger = logging.getLogger(path=f"{path}/callback_log.log")

    def on_acquisition_maximized(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        def extact_data_list():
            acquisition_function_values = acquisition_function(configs)[:, 0]
            columns = configs[0].keys()
            data = zip(*zip(*map(lambda config: config.values(), configs)), acquisition_function_values)
            return pd.DataFrame(data, columns=columns + ["acquisition_function_value"])

        self.logger.info(f"Writing data {self.counter} to csv")
        config_space: ConfigurationSpace = deepcopy(acquisition_function.model._configspace)
        config_space.seed(self.counter)
        configs = config_space.sample_configuration(self.n_configs)

        df = extact_data_list()
        df.to_csv(f"{self.path}/{self.counter}.csv")
        self.logger.info(f"Finished writing data {self.counter} to csv")
        self.counter += 1


ACQUISITION_FUNCTIONS = {
    "LCB": LCB(),
    "EI": EI(),
    "PI": PI(),
}


@hydra.main(config_path="./hydra_config", config_name="base", version_base="1.1")
def main(cfg: DictConfig):
    if "multirun" in os.getcwd():
        os.chdir("/scratch/hpc-prf-intexml/fehring/generate-smac-runs/symbolic-explanations-data-generation")

    approach = cfg["data-generation"]
    run_type = approach["run_type"]
    max_hp_comb = approach["max_hp_comb"]  # amount of hyperparameter combinations we evalute in one dataset, model combiantion
    init_design_max_ratio = approach["init_design_max_ratio"]
    init_design_n_configs_per_hyperparamter = approach["init_design_n_configs_per_hyperparamter"]
    n_optimized_params = approach["n_optimized_params"]
    n_samples = approach["n_samples"]
    job_id = approach["job_id"]
    acquisition_function = ACQUISITION_FUNCTIONS[approach["acquisition_function"]]
    seed = int(approach["seed"])
    sampling_dir_name = "runs_sampling_hpobench"
    n_configs = approach["n_configs"]  # TODO rename - amount of configs sampled randomly

    run_conf = get_run_config(n_optimized_params, max_hp_comb, job_id)  # TODO: 365 jobsids
    task_dict = get_task_dict()

    data_set_postfix = f"_{task_dict[run_conf['task_id']]}"
    optimized_parameters = list(run_conf["hp_conf"])
    model_name = get_benchmark_dict()[run_conf["benchmark"]]
    b = run_conf["benchmark"](task_id=run_conf["task_id"], hyperparameters=optimized_parameters)

    def optimization_function_wrapper(cfg, seed):
        """Helper-function: simple wrapper to use the benchmark with smac"""
        result_dict = b.objective_function(cfg, rng=seed)
        return result_dict["function_value"]

    run_name = f"{model_name.replace(' ', '_')}_{'_'.join(optimized_parameters)}{data_set_postfix}"

    sampling_dir = f"results/{sampling_dir_name}/{run_type}"
    sampling_run_dir = f"{sampling_dir}/{run_name}"
    if os.path.exists(sampling_run_dir):
        shutil.rmtree(sampling_run_dir)

    os.makedirs(sampling_run_dir)

    # setup logging
    logger = get_logger(filename=f"{sampling_run_dir}/sampling_log.log")

    logger.info(f"Start {run_type} sampling for {run_name}.")

    logger.info(f"Run: {run_name}")
    logger.info(f"Start run to sample {n_samples} samples.")

    np.random.seed(seed)

    # add only parameters to be optimized to configspace
    cs = b.get_configuration_space(seed=seed, hyperparameters=optimized_parameters)

    # path to save files to
    path = f"data/acquisition_function_data/{run_name}/{acquisition_function.__class__.__name__}/{seed}"
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/configspace.json", "w") as fh:
        fh.write(str(cs))

    callback = CustomCallback(n_configs, path)

    _, _, _ = run_smac_optimization(
        configspace=cs,
        facade=BlackBoxFacade,  # Stndard gaussian process no multi fidelity
        acquisition_function=acquisition_function,
        target_function=optimization_function_wrapper,
        n_eval=n_samples,
        run_dir=sampling_run_dir,
        seed=seed,
        n_configs_per_hyperparamter=init_design_n_configs_per_hyperparamter,
        max_ratio=init_design_max_ratio,
        callback=callback,
    )


if __name__ == "__main__":
    main()

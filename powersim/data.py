"""Compatibility imports for dataset generation and replay."""
from .instance_generation.data import (DatasetConfig, RenewableConfig, generate_dataset, installed_capacity,
                                       load_dataset, load_dataset_attempt, load_dataset_instance, package_versions,
                                       scenario_design, solar_capacity_factor, split_record_file, valid_label,
                                       wind_capacity_factor, write_json)

__all__ = ["DatasetConfig", "RenewableConfig", "generate_dataset", "installed_capacity", "load_dataset",
           "load_dataset_attempt", "load_dataset_instance", "package_versions", "scenario_design",
           "solar_capacity_factor", "split_record_file", "valid_label", "wind_capacity_factor", "write_json"]

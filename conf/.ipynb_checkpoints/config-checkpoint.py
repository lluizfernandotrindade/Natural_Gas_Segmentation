from dataclasses import dataclass


@dataclass
class RegularExpressions:
    expr_regular_match_filenames_2d: str
    expr_regular_match_filenames_3d_sgy: str
    expr_regular_match_filenames_3d_json: str


@dataclass
class GasConfig:
    n_spacing_pixels_gas: int


@dataclass
class RoiConfig:
    roi_size: int
    n_spacing_pixels_roi: int
    roi_separation_in_percentage: float
    n_traces_block: int
    random_displacement: bool


@dataclass
class ModelConfig:
    n_epochs: int
    n_units_lstm_layers: list
    n_filters_dcunet: int
    batch_size: int
    learning_rate: float
    SHUFFLE_BUFFER_SIZE: int
    concatenate_train_val_sequences: bool
    validation_split_proportion: float
    class_weights_crossentropy: list
    cross_validation_folds: int
    augmentation: bool


@dataclass
class SeedConfig:
    SEED: int


@dataclass
class FilePathConfig:
    train_json: str
    base_dir: str
    pretrained_weights: str
    campos_base_dir: str


@dataclass
class OutputPaths:
    tensorboard: str
    out_directory: str


@dataclass
class Configuration:
    regularexpression: RegularExpressions
    gas_config: GasConfig
    roi_config: RoiConfig
    model_config: ModelConfig
    seed_config: SeedConfig
    file_path_config: FilePathConfig
    output_paths: OutputPaths

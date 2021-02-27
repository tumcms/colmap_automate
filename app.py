import argparse
from os import path, listdir, makedirs
from pathlib import Path
from shutil import rmtree
from subprocess import call
from string import Template
from inspect import stack

File2Commands = {
    "1_extraction": "feature_extractor",
    "2_exaustive_matching": "exhaustive_matcher",
    "2_1_spatial_matcher": "spatial_matcher",
    "2_2_transitive_matcher": "transitive_matcher",
    "3_mapper": "mapper",
    "4_bundle_adjustment": "bundle_adjuster",
    "5_model_aligner": "model_aligner",
    "6_image_undistorter": "image_undistorter",
    "7_patch_match": "patch_match_stereo",
    "8_stereo_fusion": "stereo_fusion"
}


class ReconstructionConfig:
    def __init__(self, database_path, image_path, sparse_model_path, dense_model_path, ply_output,
                 image_global_list, logging_path, min_depth, max_depth, gps_available=False):
        self.gps_available = gps_available
        self.database_path = Path(database_path)
        self.image_path = Path(image_path)
        self.sparse_model_path = Path(sparse_model_path)
        self.dense_model_path = Path(dense_model_path)
        self.ply_output_path = Path(ply_output)
        self.sparse_model_path_mapper_out = self.sparse_model_path / "tmp"
        self.sparse_model_path_ba_in = self.sparse_model_path / "tmp" / "0"
        self.sparse_model_path_ba_out = self.sparse_model_path / "ba" if image_global_list else self.sparse_model_path
        self.image_global_list = Path(image_global_list)
        self.logging_path = Path(logging_path)
        self.min_depth = min_depth
        self.max_depth = max_depth

    @staticmethod
    def CreateStandardConfig(root_dir, image_path="", database_path="", logging_path="", min_depth=-1, max_depth=-1):
        root_dir = Path(root_dir)
        db_path = database_path if database_path else root_dir / "database.db"
        image_path = image_path if image_path else root_dir / "images"
        sparse_path = root_dir / "sparse"
        dense_path = root_dir / "dense"
        ply_output = Path(root_dir, root_dir.name + ".ply")
        logging_path = logging_path if logging_path else root_dir / "log"
        return ReconstructionConfig(db_path, image_path, sparse_path, dense_path, ply_output, "",
                                    logging_path, min_depth, max_depth)

    @classmethod
    def FromDict(cls, data):
        _class = cls(data["database_path"], data["image_path"], data["sparse_model_path"],
                     data["dense_model_path"], data["ply_output_path"], data["image_global_list"],
                     data["logging_path"], data["min_depth"], data["max_depth"], data["gps_available"])
        return _class

    def to_dict(self):
        d = self.__dict__
        d["type"] = self.__class__.__name__
        return d


def CreateDirectory(needed_path: Path):
    needed_path.mkdir(parents=True, exist_ok=True)


class Reconstructor:

    @staticmethod
    def GetPathOfCurrentFile():
        return Path(stack()[1][1])

    @staticmethod
    def Generic2SpecificJobFiles(source, dest, config: ReconstructionConfig):
        dest = Path(dest)
        source = Path(source)

        CreateDirectory(dest)

        for name in sorted([f for f in listdir(source) if not path.isdir(f)]):
            with open(source / name, "r+") as f:
                data = f.read()
                d = Template(data)

                _str = d.substitute(database_path=config.database_path,
                                    image_path=config.image_path,
                                    sparse_model_path_mapper_out=config.sparse_model_path_mapper_out,
                                    sparse_model_path_ba_in=config.sparse_model_path_ba_in,
                                    sparse_model_path_ba_out=config.sparse_model_path_ba_out,
                                    ref_image_list=config.image_global_list,
                                    sparse_model_path=config.sparse_model_path,
                                    dense_model_path=config.dense_model_path,
                                    ply_output_path=config.ply_output_path,
                                    min_depth=-1,
                                    max_depth=-1)

                with open(dest / name, "w+") as t:
                    t.write(_str)

    @staticmethod
    def execute_job(source: Path, config: ReconstructionConfig):
        task = source.name
        if task == "3_mapper":
            CreateDirectory(config.sparse_model_path_mapper_out)
        elif task == "4_bundle_adjustment":
            CreateDirectory(config.sparse_model_path_ba_out)
        elif task == "5_model_aligner":
            CreateDirectory(config.sparse_model_path)
        elif task == "6_image_undistorter":
            CreateDirectory(config.dense_model_path)

        command = File2Commands[task]
        options = ["colmap", command, "--project_path", str(source.absolute())]

        CreateDirectory(config.logging_path)
        with open(path.join(config.logging_path, task + ".log"), "wb") as log:
            call(options, stdout=log)

    @staticmethod
    def execute_all(project_folder, config: ReconstructionConfig):
        for task in sorted(listdir(project_folder)):
            file_path = Path(project_folder, task)
            Reconstructor.execute_job(file_path, config)

    @staticmethod
    def CleanUp(config: ReconstructionConfig, dense=True, mapper=True, bundle_adjustment=True):
        if dense:
            rmtree(config.dense_model_path)
        if mapper:
            rmtree(config.sparse_model_path)
        if bundle_adjustment and config.sparse_model_path_ba_out.exists():
            rmtree(config.sparse_model_path_ba_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstructor Script for Colmap. Multi-Folder & Init-Parsing", add_help=True)
    parser.add_argument('--project_dir', action="store", dest="project_directory", help="The path to the colmap project directory", default="./")
    parser.add_argument('--confdir', action="store", dest="config_directory", help="The path to the configs directory", default="")
    parser.add_argument('--ecef_info', action="store", dest="ecef_data", help="File that holds the global coordinates of the recording positions", default="")
    parser.add_argument('--plyoutdir', action="store", dest="output_directory", help="The path to the output folder", default="")
    args = parser.parse_args()

    project_directory = Path(args.project_directory).expanduser()
    reconstruction_configuration = ReconstructionConfig.CreateStandardConfig(project_directory)
    reconstruction_configuration.image_global_list = Path(args.ecef_data).expanduser() if args.ecef_data else ""
    if args.output_directory:
        reconstruction_configuration.ply_output_path = Path(args.output_directory).expanduser()
    config_target = project_directory / "tconfig"

    # dir_of_this_file = Path(stack()[0][1])
    dir_of_this_file = Reconstructor.GetPathOfCurrentFile()
    config_source = Path(args.config_directory if args.config_directory else dir_of_this_file.parent / "tconfig")

    Reconstructor.Generic2SpecificJobFiles(config_source, config_target, reconstruction_configuration)
    Reconstructor.execute_all(config_target, reconstruction_configuration)
    Reconstructor.CleanUp(reconstruction_configuration, dense=False)

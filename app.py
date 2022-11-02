import argparse
import json
from os import path, listdir, makedirs
from pathlib import Path
from shutil import rmtree
from subprocess import call
from string import Template
from inspect import stack

exec_path = Path("colmap")

File2Commands = {
    "1_extraction": "feature_extractor",
    "2_exaustive_matching": "exhaustive_matcher",
    "2_1_spatial_matcher": "spatial_matcher",
    "2_2_transitive_matcher": "transitive_matcher",
    "3_mapper": "mapper",
    "4_bundle_adjustment": "bundle_adjuster",
    "4_1_rig_bundle_adjuster": "rig_bundle_adjuster",
    "5_model_aligner": "model_aligner",
    "6_image_undistorter": "image_undistorter",
    "7_patch_match": "patch_match_stereo",
    "8_stereo_fusion": "stereo_fusion"
}


class ReconstructionEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Path):
            return str(o)
        return json.JSONEncoder.default(self, o)


class ReconstructionConfig:
    def __init__(self, database_path, image_path, sparse_model_path, dense_model_path, ply_output,
                 image_global_list, logging_path, min_depth, max_depth, match_list_path, rig_config_path, gps_available=False, ):
        self.gps_available = gps_available
        self.database_path = Path(database_path)
        self.image_path = Path(image_path)
        self.sparse_model_path = Path(sparse_model_path)
        self.dense_model_path = Path(dense_model_path)
        self.ply_output_path = Path(ply_output)
        self.sparse_model_path_mapper_out = self.sparse_model_path / "tmp"
        self.sparse_model_path_ba_in = self.sparse_model_path / "tmp" / "0"
        self.sparse_model_path_ba_out = self.sparse_model_path / "ba" if image_global_list else self.sparse_model_path
        self.rig_config_path = Path(rig_config_path)
        self.image_global_list = Path(image_global_list)
        self.logging_path = Path(logging_path)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.force_overwrite = False
        self.match_list_path = match_list_path

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as fp:
            json.dump(self.to_dict(), fp, indent=4, sort_keys=True, cls=ReconstructionEncoder)

    @classmethod
    def load(cls, config_json: Path):
        with open(config_json, 'r') as fp:
            data = json.load(fp)
        return cls.from_dict(data)

    # previously CreateStandardConfig
    @staticmethod
    def default_config(root_dir, image_path="", database_path="", sparse_path="", dense_path="", logging_path="",  min_depth=-1, max_depth=-1):
        root_dir = Path(root_dir)
        db_path = database_path if database_path else root_dir / "database.db"
        image_path = image_path if image_path else root_dir / "images"
        sparse_path = sparse_path if sparse_path else root_dir / "sparse"
        dense_path = dense_path if dense_path else root_dir / "dense"
        ply_output = Path(root_dir, root_dir.name + ".ply")
        logging_path = logging_path if logging_path else root_dir / "log"
        match_list_path = ""
        return ReconstructionConfig(db_path, image_path, sparse_path, dense_path, ply_output, "",
                                    logging_path, min_depth, max_depth, match_list_path, "")

    # previously FromDict
    @classmethod
    def from_dict(cls, data):
        _class = cls(data["database_path"], data["image_path"], data["sparse_model_path"],
                     data["dense_model_path"], data["ply_output_path"], data["image_global_list"],
                     data["logging_path"], data["min_depth"], data["max_depth"], data["match_list_path"],
                     data["rig_config_path"], data["gps_available"])
        _class.__dict__ = data
        return _class

    def to_dict(self):
        d = self.__dict__
        d["type"] = self.__class__.__name__
        return d

    def filter_configurations(path: Path):
        files = [p for p in path.glob("*") if p.suffix == ""]
        return files


def CreateDirectory(needed_path: Path):
    Path(needed_path).mkdir(parents=True, exist_ok=True)


class Reconstructor:

    @staticmethod
    def GetPathOfCurrentFile():
        return Path(stack()[1][1])

    @staticmethod
    def Generic2SpecificJobFiles(source, dest, config: ReconstructionConfig):
        dest = Path(dest)
        source = Path(source)
        CreateDirectory(dest)

        for src in sorted(ReconstructionConfig.filter_configurations(source)):
            with open(src, "r+") as f:
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
                                    match_list_path=config.match_list_path,
                                    rig_config_path=config.rig_config_path,
                                    min_depth=-1,
                                    max_depth=-1)

                dest_path = dest / src.name
                if not dest_path.exists() or config.force_overwrite:
                    with open(dest_path, "w+") as t:
                        t.write(_str)

    @staticmethod
    def execute_job(source: Path, config: ReconstructionConfig):
        task = source.name
        if task == "3_mapper":
            CreateDirectory(config.sparse_model_path_mapper_out)
        elif task == "4_bundle_adjustment" or task == "4_1_rig_bundle_adjuster":
            CreateDirectory(config.sparse_model_path_ba_out)
        elif task == "5_model_aligner":
            CreateDirectory(config.sparse_model_path)
        elif task == "6_image_undistorter":
            CreateDirectory(config.dense_model_path)

        command = File2Commands[task]
        options = [str(exec_path), command, "--project_path", str(source.absolute())]

        CreateDirectory(config.logging_path)
        with open(path.join(config.logging_path, task + ".log"), "wb") as log:
            call(options, stdout=log)

    @staticmethod
    def execute_all(config_folder, config: ReconstructionConfig):
        for file_path in sorted(ReconstructionConfig.filter_configurations(config_folder)):
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
    parser.add_argument("project_path", nargs=1, type=Path,
                        help="The path to reconstructer job. If no config is present, a default config will be initialized", default=None)
    parser.add_argument('-exec', "--execution_command", type=Path,
                        help="Add path to the CMA config to run the job. If left empty is specified all jobs are executed", default=None,
                        nargs='?', const='switch_flag')
    # parser.add_argument('--ecef_info', action="store", dest="ecef_data", help="File that holds the global coordinates of the recording positions", default="")
    parser.add_argument("-cfg", "--config_name", help="If you need multi-config support", default="config")
    parser.add_argument("-tmpl", "--template_path", help="Path to custom templates", type=Path, default=Path(__file__).absolute().parent / "tconfig")
    args = parser.parse_args()

    project_path = args.project_path[0].expanduser()

    if project_path.exists():
        config_folder = project_path / args.config_name
        rec_config_path = config_folder / f"{args.config_name}.json"

        if not rec_config_path.exists():
            reconstruction_configuration = ReconstructionConfig.default_config(project_path)
            reconstruction_configuration.save(rec_config_path)
        else:
            reconstruction_configuration = ReconstructionConfig.load(rec_config_path)

        cmd = args.execution_command
        if cmd:
            if cmd == "switch_flag":
                Reconstructor.execute_all(config_folder, reconstruction_configuration)
                Reconstructor.CleanUp(reconstruction_configuration, dense=False)
            else:
                if str(cmd).find("/") == -1:
                    cmd = config_folder / cmd
                Reconstructor.execute_job(Path(cmd), reconstruction_configuration)

        else:
            Reconstructor.Generic2SpecificJobFiles(args.template_path, config_folder, reconstruction_configuration)

    else:
        raise FileNotFoundError("Please provide a folder, with an adherent base_strucutre")

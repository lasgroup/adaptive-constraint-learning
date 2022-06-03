import os
from datetime import datetime

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import numpy as np

jsonpickle_numpy.register_handlers()


class Artifact:
    def __init__(self, file_name, method, _run):
        self._run = _run
        self.file_name = file_name
        self.file_path = os.path.join("/tmp", file_name)
        if method is not None:
            self.file_obj = open(self.file_path, method)
        else:
            self.file_obj = None

    def __enter__(self):
        if self.file_obj is None:
            return self.file_path
        else:
            return self.file_obj

    def __exit__(self, type, value, traceback):
        if self.file_obj is not None:
            self.file_obj.close()
        self._run.add_artifact(self.file_path)


class FileExperimentResults:
    def __init__(self, result_folder):
        self.result_folder = result_folder
        self.config = self._read_json(result_folder, "config.json")
        self.metrics = self._read_json(result_folder, "metrics.json")
        self.run = self._read_json(result_folder, "run.json")
        self.info = self._read_json(result_folder, "info.json")
        self.status = self.run["status"]
        self.result = self.run["result"]

    def _read_json(self, result_folder, filename):
        try:
            with open(os.path.join(result_folder, filename), "r") as f:
                json_str = f.read()
            return jsonpickle.loads(json_str)
        except Exception:
            return None

    def _convert_str_to_time(self, time_str):
        try:
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S.%f")
        except ValueError:
            return datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%S")

    def _get_start_time(self):
        return self._convert_str_to_time(self.run["start_time"])

    def _get_stop_time(self):
        return self._convert_str_to_time(self.run["stop_time"])

    def get_metric(self, name, get_runtime=False):
        metric = self.metrics[name]
        if get_runtime:
            start_time = self._get_start_time()
            timestamps = [self._convert_str_to_time(t) for t in metric["timestamps"]]
            run_time = [(t - start_time).total_seconds() for t in timestamps]
            return metric["steps"], metric["values"], run_time
        else:
            return metric["steps"], metric["values"]

    def print_captured_output(self):
        with open(os.path.join(self.result_folder, "cout.txt"), "r") as f:
            print(f.read())

    def get_runtime(self):
        start_time = self._get_start_time()
        stop_time = self._get_stop_time()
        return (stop_time - start_time).total_seconds()

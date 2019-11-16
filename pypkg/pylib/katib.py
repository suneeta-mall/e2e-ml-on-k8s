import json
import logging
import os
import subprocess
import tempfile
import time

import schedule
from pylib import execute_cmd
from retrying import retry

from .utils import execute_cmd


# http://localhost:8080/_/katib/?ns=kubeflow-anonymous
# https://raw.githubusercontent.com/kubeflow/katib/master/examples/v1alpha2/hyperband-example.yaml

class KatibController:
    def __init__(self, version, commit_id, base_input_dir, input_version):
        self.version = version
        self.commit_id = commit_id
        self.input = base_input_dir
        self.input_version = input_version

        self._tag = 'check_experiment'
        _work_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(_work_dir, "templates", "katib-hp-tunning.yaml")) as f:
            self._katib_template = f.read()

    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=4)
    def create_katib_experiment(self):
        with tempfile.NamedTemporaryFile(mode='w') as fp:
            katib_yaml = self._katib_template.format(short_commit_id=self.commit_id[0:6], commit_id=self.commit_id,
                                                     version=self.version,
                                                     input=self.input, input_version=self.input_version)
            fp.write(katib_yaml)
            fp.seek(0)
            execute_cmd(['kubectl', 'apply', '-f', fp.name])
            logging.info("Tunning job created!")

    def schedule(self, poll_freq):
        @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=4)
        def check_experiment(commit_id, tag):
            logging.debug(f"Checking on schedule if experiment completed")
            # kubectl -n kubeflow get experiments --selector=app=FLAGS.version \
            # -o=jsonpath='{.items[?(@.status.completionTime)].metadata.name}'
            cmd_arr = ['kubectl', 'get', 'experiments', '-n', 'kubeflow', f'--selector=app={commit_id}',
                       "-o=jsonpath='{.items[?(@.status.completionTime)].metadata.name}'"]
            logging.info(f"Running command {' '.join(cmd_arr)} checking for experiments")
            process_res = execute_cmd(cmd_arr)
            completed = process_res == f"'tune-{commit_id[0:6]}'"
            if completed:
                schedule.clear(tag)

        job = schedule.every(poll_freq).minutes.do(check_experiment, self.commit_id, self._tag).tag(self._tag)
        experiment_running = True
        while experiment_running:
            schedule.run_pending()
            experiment_running = bool(schedule.default_scheduler.jobs)
            time.sleep(int(poll_freq / 2))
        logging.info("Finished running experiment")

    @retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=4)
    def get_hp_params(self):
        logging.debug(f"Collecting results of experiment")
        cmd = f"kubectl get experiments -n kubeflow --selector=app={self.commit_id} -o json \
        | jq '.items[]|{{assignments:.status.currentOptimalTrial.parameterAssignments," \
              f"observation:.status.currentOptimalTrial.observation}}'"
        logging.info(f"Running command {cmd}")
        ps = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        process_res = ps.communicate()[0]
        hp_payload = json.loads(process_res.decode('utf-8'))
        hp = dict()
        for param_set in hp_payload['assignments']:
            name = param_set['name'].replace("--", "")
            val = param_set['value']
            if name in ['batch_size', 'epochs', 'steps_per_epoch']:
                hp[name] = int(val)
            elif name == 'learning_rate':
                hp[name] = float(val)
            else:
                hp[name] = str(val)
        logging.info(f"Returned hp are: {hp_payload}")
        return hp

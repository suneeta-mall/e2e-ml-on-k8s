import argparse
import json
import os

from pylib import set_log_level, KatibController

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", help="Base path of data source input", type=str, required=True)
    parser.add_argument('--input_version', help='Version of container to run for tunning', type=str, default=None)
    parser.add_argument("--output", help="Base path of dataset output", type=str, required=True)

    tip = "Whether to achieve full or close reproducibility"
    parser.add_argument("--fully-reproducible", help=tip, dest='reproducible', action='store_true')
    parser.add_argument('--closely-reproducible', help=tip, dest='reproducible', action='store_false')
    parser.set_defaults(reproducible=True)

    parser.add_argument('--container_version', help='Version of container to run for tunning', type=str, default=None)
    parser.add_argument('--process_id', help='ID to be used for process', type=str,
                        default=os.getenv("PACH_OUTPUT_COMMIT_ID", "latest"))

    parser.add_argument("--poll_frequency", type=int, default=10,
                        help="Frequency in minutes at which scheduler should check for experiment status")

    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])

    FLAGS = parser.parse_args()
    set_log_level(FLAGS.log.upper())

    if not FLAGS.container_version:
        with open("/app/version") as f:
            FLAGS.container_version = f.read()

    if not FLAGS.input_version:
        repo = FLAGS.input.split(os.sep)[2]
        FLAGS.input_version = os.getenv(f"{repo}_COMMIT", "master")

    katib = KatibController(FLAGS.container_version, FLAGS.process_id, FLAGS.input, FLAGS.input_version)
    katib.create_katib_experiment()
    katib.schedule(FLAGS.poll_frequency)

    best_params = katib.get_hp_params()
    with open(os.path.join(FLAGS.output, "optimal_hp.json"), 'w') as f:
        json.dump(best_params, f)

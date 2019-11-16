import logging
import os
import tempfile

from .utils import execute_cmd


class SeldonController:
    def __init__(self, container_version, model_db, version):
        self.version = version
        self.model_db = model_db
        self.container_version = container_version
        _work_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(_work_dir, "templates", "model-serving.yaml")) as f:
            self._seldon_template = f.read()

    def _serving_action(self, action="apply"):
        with tempfile.NamedTemporaryFile(mode='w') as fp:
            seldon_yaml = self._seldon_template.format(short_version=self.version[0:6],
                                                       version=self.version,
                                                       model_db=self.model_db,
                                                       container_version=self.container_version)
            fp.write(seldon_yaml)
            fp.seek(0)
            execute_cmd(['kubectl', action, '-f', fp.name])
            logging.info("Seldon serving service is action %s is performed!", action)

    def create_serving(self):
        self._serving_action(action='apply')

    def delete_serving(self):
        self._serving_action(action='delete')

#!/usr/bin/env python

import argparse
import base64
import logging
import os
import shutil
import tarfile
import tempfile
import urllib.request
from io import StringIO
from pathlib import Path

import pandas as pd
from pylib import set_log_level


# Open source dataset by Oxford Uni. ->  The Oxford-IIIT Pet Dataset
# https://www.robots.ox.ac.uk/~vgg/data/pets/
# https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz


def download_petset_data(base_dir: str) -> list:
    img_tar_fn = os.path.join(base_dir, 'images.tar.gz')
    mask_tar_fn = os.path.join(base_dir, 'annotations.tar.gz')

    urllib.request.urlretrieve("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
                               img_tar_fn)
    logging.info('PetSet image tar downloaded')
    urllib.request.urlretrieve("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
                               mask_tar_fn)
    logging.info('PetSet segmentation mask tar downloaded')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Base path of dataset output", type=str, default=None)
    parser.add_argument("--output", help="Base path of dataset output", type=str, default='/pfs/out')
    parser.add_argument("--log", help="sets the logging level", type=str, default=os.getenv("LOG_LEVEL", "info"),
                        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "critical", "error", "warning",
                                 "info", "debug"])
    FLAGS = parser.parse_args()

    set_log_level(FLAGS.log.upper())

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info('created temporary directory: %s', temp_dir)
        tar_gz_loc = temp_dir
        if FLAGS.input and os.path.exists(FLAGS.input):
            tar_gz_loc = FLAGS.input
            logging.info("Source  files are provided")
        else:
            logging.info("Download default dataset")
            download_petset_data(temp_dir)

        for tar_fn in Path(tar_gz_loc).glob("*.tar.gz"):
            with tarfile.open(tar_fn) as tar:
                tar.extractall(path=temp_dir)

        logging.info('Unpacked all tars')

        header = ["file_name", "class_id", "species", "breed"]
        with open(os.path.join(temp_dir, "annotations", "list.txt")) as f:
            # skip first 6 lines to remove comments
            lines = f.readlines()[6:]
        df = pd.read_csv(StringIO(''.join(lines)), sep=" ", header=None, names=header)

        logging.info('Creating PetSet dataset sorted by petset id: %s ', FLAGS.output)
        for petset_img in Path(temp_dir).glob('**/*.jpg'):
            fpath, ext = os.path.splitext(petset_img.as_posix())
            petset_id = os.path.basename(fpath)
            petset_group = str(petset_id.rpartition("_")[0])

            petset_mask_fn = os.path.join(temp_dir, "annotations", "trimaps", f"{petset_id}.png")

            if os.path.exists(petset_mask_fn):
                petset_out_dir = os.path.join(FLAGS.output, petset_id)
                os.makedirs(petset_out_dir, exist_ok=True)

                with open(os.path.join(petset_out_dir, 'metadata.json'), 'w') as meta_fn:
                    row = df[df.file_name == petset_id]
                    if row.shape[0] == 0:
                        logging.warning(f"No metadata found for {petset_id}, inferring from group")
                        row = df[df.file_name.str.startswith(petset_group)]
                    meta = '''
{{
    "features": {{
      "feature": {{
         "Breed": {{ "int64_list": {{ value: [{breed}] }} }},
         "ClassId": {{ "int64_list": {{ value: [{class_id}] }} }},
         "Species": {{ "int64_list": {{ value: [{species}] }} }},
         "ClassName": {{ "bytes_list": {{ value: ["{name}"] }} }}
      }}
    }}
}}
                    '''.format(breed=int(row.breed.values[0]), class_id=int(row.class_id.values[0]),
                               species=int(row.species.values[0]),
                               name=base64.b64encode(petset_group.encode()).decode())
                    meta_fn.write(meta)

                shutil.move(petset_img.as_posix(), os.path.join(petset_out_dir, "image.jpg"))
                shutil.move(petset_mask_fn, os.path.join(petset_out_dir, "mask.png"))

    logging.info('PetSet dataset created sorted by petset id')

import logging
import os

import python_pachyderm as pach
from retrying import retry


@retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fn_with_retry(fn, commit_tpl: tuple, fn_path: str):
    return fn(commit_tpl, fn_path)


class FSClient:
    def __init__(self, db, host="localhost", port=650):
        self._client = pach.Client(host=host, port=port)
        self._db = db

    def _write_content(self, file_obj, dest):
        _base_dir = os.path.dirname(dest)
        if _base_dir:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "ab") as f:
            f.write(file_obj)

    def download_files(self, version, file_pattern, dest):
        list_content_gen = fn_with_retry(self._client.glob_file, (self._db, version), file_pattern)
        for content_fn in list_content_gen:
            src_fn = content_fn.file.path
            if content_fn.file_type != pach.FileType.FILE:
                continue
            file_content_gen = fn_with_retry(self._client.get_file, (self._db, version), src_fn)
            for file_obj in file_content_gen:
                dst_fn = os.path.join(dest, src_fn[1:] if src_fn.startswith(os.sep) else src_fn)
                self._write_content(file_obj, dst_fn)
        logging.info("Finished Downloading file from %s to destination %s", self._db, dest)

    def download_file(self, version, file, dst_fn):
        content_gen = fn_with_retry(self._client.get_file, (self._db, version), file)
        for file_obj in content_gen:
            self._write_content(file_obj, dst_fn)


def download_input(base_input_dir, fn_glob="/{validation,training}/*/*.{jpg,png}",
                   commit=os.getenv("COMMIT_ID", "master"),
                   fs_host=os.getenv("FS_ADDRESS", "localhost")):
    repo = base_input_dir.split(os.sep)[2] if base_input_dir.startswith("/pfs/") else base_input_dir
    c = FSClient(repo, fs_host)
    # Download files to input dir
    c.download_files(commit, fn_glob, base_input_dir)

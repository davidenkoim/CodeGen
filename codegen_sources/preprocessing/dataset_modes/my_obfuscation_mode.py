import os
import sys
import time
from itertools import chain
from logging import getLogger
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

import tqdm
from submitit import Executor

from codegen_sources.preprocessing.dataset_modes.dataset_mode import TIMEOUT
from codegen_sources.preprocessing.dataset_modes.obfuscation_mode import ObfuscationMode
from codegen_sources.preprocessing.lang_processors.lang_processor import LangProcessor
from codegen_sources.preprocessing.obfuscation.utils_deobfuscation import SEPARATOR
from codegen_sources.preprocessing.utils import is_valid_file

logger = getLogger()

extensions = {"java": ".java", "python": ".py", "kotlin": ".kt"}


class MyObfuscationMode(ObfuscationMode):
    def extract_data_and_tokenize(
            self, executor: Executor = None, local_parallelism: int = None
    ):
        """
        Takes the root folder of the dataset, containing json files as input
        For each json in it extract data, tokenize, and save in dedicated .tok file
        """
        self.id_is_line = True
        logger.info("")
        logger.info("")
        logger.info("========== Extract and Tokenize ===========")
        if local_parallelism is not None:
            logger.info(f"Using {local_parallelism} processors.")

        files, file_langs = zip(*chain(*[self.get_files_with_extension(lang) for lang in self.languages])) \
            if any(not is_valid_file(Path(name)) for name in self.get_tok_files().values()) else ([], [])

        logger.info(
            f"{' '.join(self.languages)}: tokenizing and extracting obfuscated code from {len(files)} files ..."
        )
        if len(files) > 0:
            self.extract_from_files(files, file_langs, self.bpe.process_strings, local_parallelism)
        else:
            return logger.info("Data extraction and tokenization is already done.")

    def get_tok_files(self):
        return {
            suffix: self.folder.joinpath(f"{lang}.all.{suffix}.tok")
            for suffix in self.suffixes
            for lang in self.languages
        }

    def extract_from_files(
            self,
            input_paths: List[str],
            langs: List[str],
            process_strings: bool,
            local_parallelism: int = None,
    ):
        """
        Takes one json file as input. For each document, it extracts data and tokenizes it.
        The results are written into a .tok file.
        """
        # {suffix: open(output)}
        tok_files = self.get_tok_files()
        tok_files = self.open_tok_files(tok_files)

        lines = [(path, {}, lang, process_strings) for path, lang in zip(input_paths, langs)]

        number_errors = 0
        number_timeouts = 0
        multilines_code = 0
        number_lines = len(lines)
        logger.info(f"Number of lines to process: {number_lines}")
        filtered_examples = 0
        try:
            start = time.time()
            if local_parallelism:
                assert cpu_count() > (
                        local_parallelism - 1
                ), "Number of processors must be greater than number of max workers in ProcessPoolExecutor"
                # Leave one processor free for other tasks.
                executor = Pool(
                    processes=cpu_count() - local_parallelism - 1,
                    initializer=self.initialize_processor,
                )
            else:
                executor = Pool(
                    processes=cpu_count(), initializer=self.initialize_processor
                )
            results_for_line = list(tqdm.tqdm(
                executor.imap(self.checkpoint_line, lines), total=len(lines), mininterval=1
            ))
            # results_for_line = [self.checkpoint_line(line) for line in tqdm.tqdm(lines)]

            for line_id, repo, tokenized_data in results_for_line:
                self.processed_lines.add(line_id)
                # returning None means there was an issue
                if tokenized_data == TIMEOUT:
                    number_timeouts += 1
                    continue
                if (
                        tokenized_data is None
                        or all(v is None for v in tokenized_data.values())
                        or len(tokenized_data) == 0
                        or repo is None
                ):
                    number_errors += 1
                    continue
                if self.parallel_dataset:
                    if any(v is None for v in tokenized_data.values()):
                        number_errors += 1
                        continue
                    expected_length = len(next(iter(tokenized_data.values())))
                    if not all(
                            expected_length == len(v) for v in tokenized_data.values()
                    ):
                        number_errors += 1
                        continue
                if self.filter(tokenized_data):
                    filtered_examples += 1
                    continue
                for suffix, tok_codes in tokenized_data.items():
                    if tok_codes is None:
                        assert not self.parallel_dataset
                        number_errors += 1
                        continue
                    for tok_code in tok_codes:
                        if not len(tok_code.splitlines()) <= 1:
                            multilines_code += 1
                        try:
                            tok_files[suffix].write(repo + SEPARATOR + tok_code)
                            tok_files[suffix].write("\n")
                        except KeyboardInterrupt:
                            raise
                        except Exception:
                            sys.stderr.write(f"Exception writing data: {tok_code}\n")
                            number_errors += 1
                            continue
                for suffix, _ in tokenized_data.items():
                    tok_files[suffix].flush()
            end = time.time()
            logger.info(f"Time elapsed: {round((end - start), 2)}")
            if number_errors > 0:
                logger.warning(
                    f"Tokenization of {self.folder}:"
                    f"{number_errors} errors out of {number_lines} lines"
                    f"({number_errors / number_lines:.2%})"
                )
            if number_timeouts > 0:
                logger.warning(
                    f"Tokenization of {self.folder}:"
                    f"{number_timeouts} timeouts out of {number_lines} lines"
                    f"({number_timeouts / number_lines:.2%})"
                )

            if filtered_examples > 0:
                logger.warning(
                    f"Tokenization of {self.folder}:"
                    f"{filtered_examples} filtered examples in {number_lines} lines"
                    f"({filtered_examples / number_lines:.2%})"
                )
            if multilines_code > 0:
                logger.warning(
                    f"Tokenization of {self.folder}:"
                    f"{multilines_code} multiline codes {number_lines} lines"
                    f"({multilines_code / number_lines:.2%})"
                )
        except TimeoutError:
            # The tokenization process is sometimes killed and it makes the multiprocessing hang forever
            for f in tok_files.values():
                f.close()
            logger.warning("Program closed automatically after one hour")
            exit(1)

    def extract_data_for_line(self, line_id, json_line: dict, process_strings: bool,
                              lang_processor: LangProcessor):
        return super().extract_data_for_line(line_id, self.get_content(line_id), process_strings, lang_processor)

    def get_content(self, path):
        return {"content": read_file(path),
                "repo_name": os.path.normpath(os.path.relpath(path, self.folder)).split(os.sep)[0]}

    def get_files_with_extension(self, lang):
        return [(file, lang) for file in self.folder.rglob(f"*{extensions[lang]}") if file.is_file()]


def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

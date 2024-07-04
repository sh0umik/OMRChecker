"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""
import os
from csv import QUOTE_NONNUMERIC
from pathlib import Path
from time import time

import cv2
import pandas as pd
from src import constants
from src.evaluation import evaluate_concatenated_response
from src.logger import logger
from src.template import Template
from src.utils.image import ImageUtils
from src.utils.interaction import InteractionUtils, Stats
from src.utils.parsing import get_concatenated_response, open_config_with_defaults

from src.utils.file import Paths, setup_outputs_for_template

# Load processors
STATS = Stats()


def process_and_get_result(
        file_data,
        file_name,
        outputs_namespace,
):

    in_omr = cv2.imdecode(file_data, cv2.IMREAD_GRAYSCALE)

    logger.info("")
    logger.info(
        f"Opening image: \t'{file_name}'\tResolution: {in_omr.shape}"
    )

    curr_dir = Path()

    # Update local tuning_config (in current recursion stack)
    local_config_path = curr_dir.joinpath(constants.CONFIG_FILENAME)
    tuning_config = open_config_with_defaults(local_config_path)

    # Update local template (in current recursion stack)
    local_template_path = curr_dir.joinpath(constants.TEMPLATE_FILENAME)
    template = Template(
        local_template_path,
        tuning_config,
    )

    template.image_instance_ops.reset_all_save_img()

    template.image_instance_ops.append_save_img(1, in_omr)

    in_omr = template.image_instance_ops.apply_preprocessors(
        file_name, in_omr, template
    )

    output_dir = Path('output')
    paths = Paths(output_dir)

    outputs_namespace = setup_outputs_for_template(paths, template)

    if in_omr is None:
        # Error OMR case
        new_file_path = outputs_namespace.paths.errors_dir.joinpath(file_name)
        outputs_namespace.OUTPUT_SET.append(
            [file_name] + outputs_namespace.empty_resp
        )
        if check_and_move(
                constants.ERROR_CODES.NO_MARKER_ERR, file_name, new_file_path
        ):
            err_line = [
                           file_name,
                           file_name,
                           new_file_path,
                           "NA",
                       ] + outputs_namespace.empty_resp
            pd.DataFrame(err_line, dtype=str).T.to_csv(
                outputs_namespace.files_obj["Errors"],
                mode="a",
                quoting=QUOTE_NONNUMERIC,
                header=False,
                index=False,
            )
            return

    # uniquify
    file_id = str(file_name)
    save_dir = outputs_namespace.paths.save_marked_dir
    (
        response_dict,
        final_marked,
        multi_marked,
        _,
    ) = template.image_instance_ops.read_omr_response(
        template, image=in_omr, name=file_id, save_dir=save_dir
    )

    # TODO: move inner try catch here
    # concatenate roll nos, set unmarked responses, etc
    omr_response = get_concatenated_response(response_dict, template)

    return omr_response


def check_and_move(error_code, file_path, filepath2):
    # TODO: fix file movement into error/multimarked/invalid etc again
    STATS.files_not_moved += 1
    return True


def print_stats(start_time, files_counter, tuning_config):
    time_checking = max(1, round(time() - start_time, 2))
    log = logger.info
    log("")
    log(f"{'Total file(s) moved':<27}: {STATS.files_moved}")
    log(f"{'Total file(s) not moved':<27}: {STATS.files_not_moved}")
    log("--------------------------------")
    log(
        f"{'Total file(s) processed':<27}: {files_counter} ({'Sum Tallied!' if files_counter == (STATS.files_moved + STATS.files_not_moved) else 'Not Tallying!'})"
    )

    if tuning_config.outputs.show_image_level <= 0:
        log(
            f"\nFinished Checking {files_counter} file(s) in {round(time_checking, 1)} seconds i.e. ~{round(time_checking / 60, 1)} minute(s)."
        )
        log(
            f"{'OMR Processing Rate':<27}:\t ~ {round(time_checking / files_counter, 2)} seconds/OMR"
        )
        log(
            f"{'OMR Processing Speed':<27}:\t ~ {round((files_counter * 60) / time_checking, 2)} OMRs/minute"
        )
    else:
        log(f"\n{'Total script time':<27}: {time_checking} seconds")

    if tuning_config.outputs.show_image_level <= 1:
        log(
            "\nTip: To see some awesome visuals, open config.json and increase 'show_image_level'"
        )

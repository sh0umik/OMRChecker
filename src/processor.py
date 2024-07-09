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

from src.entry import check_and_move

# Load processors
STATS = Stats()


def process_and_get_result(
        template_id,
        file_data,
        file_name,
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

    if template_id is None:
        template_path = constants.TEMPLATE_FILENAME
    else:
        template_path = "templates/" + template_id + ".json"

    # Update local template (in current recursion stack)
    local_template_path = curr_dir.joinpath(template_path)
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

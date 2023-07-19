from src.constants import FIELD_TYPES
from src.schemas.constants import ARRAY_OF_STRINGS, FIELD_STRING_TYPE

positive_number = {"type": "number", "minimum": 0}
positive_integer = {"type": "integer", "minimum": 0}
two_positive_integers = {
    "type": "array",
    "prefixItems": [
        positive_integer,
        positive_integer,
    ],
    "maxItems": 2,
    "minItems": 2,
}
two_positive_numbers = {
    "type": "array",
    "prefixItems": [
        positive_number,
        positive_number,
    ],
    "maxItems": 2,
    "minItems": 2,
}
zero_to_one_number = {
    "type": "number",
    "minimum": 0,
    "maximum": 1,
}
patch_area_description = {
    "type": "object",
    "required": ["origin", "dimensions", "margins"],
    "additionalProperties": False,
    "properties": {
        "origin": two_positive_integers,
        "dimensions": two_positive_integers,
        "margins": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "vertical": positive_integer,
                "horizontal": positive_integer,
            },
        },
        "pointsSelector": {
            "type": "string",
            "enum": [
                "DOT_TOP_LEFT",
                "DOT_TOP_RIGHT",
                "DOT_BOTTOM_RIGHT",
                "DOT_BOTTOM_LEFT",
                "DOT_CENTER",
                "LINE_INNER_EDGE",
                "LINE_OUTER_EDGE",
            ],
        },
    },
}

# if_requirements help in suppressing redundant errors from 'allOf'
pre_processor_if_requirements = {
    "required": ["name", "options"],
}
crop_on_markers_if_requirements = {
    "required": ["type"],
}
pre_processor_options_available_keys = {"processingDimensions": True}

crop_on_markers_tuning_options_available_keys = {
    "dotKernel": True,
    "lineKernel": True,
    "apply_erode_subtract": True,
    "marker_rescale_range": True,
    "marker_rescale_steps": True,
    "max_matching_variation": True,
    "min_matching_threshold": True,
}
crop_on_markers_options_available_keys = {
    **pre_processor_options_available_keys,
    "pointsSelector": True,
    "tuningOptions": True,
    "type": True,
}

crop_on_dot_lines_tuning_options = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "dotKernel": two_positive_integers,
        "lineKernel": two_positive_integers,
    },
}

TEMPLATE_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Udayraj123/OMRChecker/tree/master/src/schemas/template-schema.json",
    "title": "Template Validation Schema",
    "description": "OMRChecker input template schema",
    "type": "object",
    "required": [
        "bubbleDimensions",
        "pageDimensions",
        "preProcessors",
        "fieldBlocks",
    ],
    "additionalProperties": False,
    "properties": {
        "bubbleDimensions": {
            **two_positive_integers,
            "description": "The dimensions of the overlay bubble area: [width, height]",
        },
        "customLabels": {
            "description": "The customLabels contain fields that need to be joined together before generating the results sheet",
            "type": "object",
            "patternProperties": {
                "^.*$": {"type": "array", "items": FIELD_STRING_TYPE}
            },
        },
        "outputColumns": {
            "type": "array",
            "items": FIELD_STRING_TYPE,
            "description": "The ordered list of columns to be contained in the output csv(default order: alphabetical)",
        },
        "pageDimensions": {
            **two_positive_integers,
            "description": "The dimensions(width, height) to which the page will be resized to before applying template",
        },
        "preProcessors": {
            "description": "Custom configuration values to use in the template's directory",
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "AutoAlignTemplate",
                            "CropOnMarkers",
                            "CropPage",
                            "FeatureBasedAlignment",
                            "GaussianBlur",
                            "Levels",
                            "MedianBlur",
                        ],
                    },
                    "options": {
                        "type": "object",
                        "properties": {
                            "processingDimensions": two_positive_integers,
                        },
                    },
                },
                **pre_processor_if_requirements,
                "allOf": [
                    {
                        "if": {
                            "properties": {"name": {"const": "AutoAlignTemplate"}},
                            **pre_processor_if_requirements,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "match_col": {
                                            "type": "integer",
                                            "minimum": 0,
                                            "maximum": 10,
                                        },
                                        "max_steps": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 100,
                                        },
                                        "morph_threshold": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 255,
                                        },
                                        "stride": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 20,
                                        },
                                        "thickness": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": 10,
                                        },
                                    },
                                },
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "CropPage"}},
                            **pre_processor_if_requirements,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "morphKernel": two_positive_integers
                                    },
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "FeatureBasedAlignment"}},
                            **pre_processor_if_requirements,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "2d": {"type": "boolean"},
                                        "goodMatchPercent": {"type": "number"},
                                        "maxFeatures": {"type": "integer"},
                                        "reference": {"type": "string"},
                                        "matcherType": {
                                            "type": "string",
                                            "enum": [
                                                "DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING",
                                                "NORM_HAMMING",
                                            ],
                                        },
                                    },
                                    "required": ["reference"],
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "GaussianBlur"}},
                            **pre_processor_if_requirements,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "kSize": two_positive_integers,
                                        "sigmaX": {"type": "number"},
                                    },
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "Levels"}},
                            **pre_processor_if_requirements,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {
                                        "gamma": zero_to_one_number,
                                        "high": zero_to_one_number,
                                        "low": zero_to_one_number,
                                    },
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "MedianBlur"}},
                            **pre_processor_if_requirements,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "properties": {"kSize": {"type": "integer"}},
                                }
                            }
                        },
                    },
                    {
                        "if": {
                            "properties": {"name": {"const": "CropOnMarkers"}},
                            **pre_processor_if_requirements,
                        },
                        "then": {
                            "properties": {
                                "options": {
                                    "type": "object",
                                    # Note: "required" key is retrieved from crop_on_markers_if_requirements
                                    "properties": {
                                        # Note: the keys need to match with crop_on_markers_options_available_keys
                                        **crop_on_markers_options_available_keys,
                                        "pointsSelector": {
                                            "type": "string",
                                            "enum": [
                                                "CENTERS",
                                                "INNER_WIDTHS",
                                                "INNER_HEIGHTS",
                                                "INNER_CORNERS",
                                                "OUTER_CORNERS",
                                            ],
                                        },
                                        "type": {
                                            "type": "string",
                                            "enum": [
                                                "CUSTOM_MARKER",
                                                "ONE_LINE_TWO_DOTS",
                                                "ONE_LINE_TWO_DOTS_MIRROR",
                                                "TWO_LINES",
                                                "FOUR_DOTS",
                                            ],
                                        },
                                    },
                                    **crop_on_markers_if_requirements,
                                    "allOf": [
                                        {
                                            "if": {
                                                **crop_on_markers_if_requirements,
                                                "properties": {
                                                    "type": {"const": "CUSTOM_MARKER"}
                                                },
                                            },
                                            "then": {
                                                "required": [
                                                    "relativePath",
                                                    "dimensions",
                                                ],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "dimensions": two_positive_integers,
                                                    "relativePath": {"type": "string"},
                                                    "topRightDot": patch_area_description,
                                                    "bottomRightDot": patch_area_description,
                                                    "topLeftDot": patch_area_description,
                                                    "bottomLeftDot": patch_area_description,
                                                    "tuningOptions": {
                                                        "type": "object",
                                                        "additionalProperties": False,
                                                        "properties": {
                                                            "apply_erode_subtract": {
                                                                "type": "boolean"
                                                            },
                                                            # Range of rescaling in percentage -
                                                            "marker_rescale_range": two_positive_integers,
                                                            "marker_rescale_steps": positive_integer,
                                                            "max_matching_variation": {
                                                                "type": "number"
                                                            },
                                                            "min_matching_threshold": {
                                                                "type": "number"
                                                            },
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                        {
                                            "if": {
                                                **crop_on_markers_if_requirements,
                                                "properties": {
                                                    "type": {
                                                        "const": "ONE_LINE_TWO_DOTS"
                                                    }
                                                },
                                            },
                                            "then": {
                                                "required": [
                                                    "leftLine",
                                                    "topRightDot",
                                                    "bottomRightDot",
                                                ],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "tuningOptions": crop_on_dot_lines_tuning_options,
                                                    "leftLine": patch_area_description,
                                                    "topRightDot": patch_area_description,
                                                    "bottomRightDot": patch_area_description,
                                                    # TODO: add "topLeftDot": False, etc here?
                                                },
                                            },
                                        },
                                        {
                                            "if": {
                                                **crop_on_markers_if_requirements,
                                                "properties": {
                                                    "type": {
                                                        "const": "ONE_LINE_TWO_DOTS_MIRROR"
                                                    }
                                                },
                                            },
                                            "then": {
                                                "required": [
                                                    "rightLine",
                                                    "topLeftDot",
                                                    "bottomLeftDot",
                                                ],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "tuningOptions": crop_on_dot_lines_tuning_options,
                                                    "rightLine": patch_area_description,
                                                    "topLeftDot": patch_area_description,
                                                    "bottomLeftDot": patch_area_description,
                                                },
                                            },
                                        },
                                        {
                                            "if": {
                                                **crop_on_markers_if_requirements,
                                                "properties": {
                                                    "type": {"const": "TWO_LINES"}
                                                },
                                            },
                                            "then": {
                                                "required": ["leftLine", "rightLine"],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "tuningOptions": crop_on_dot_lines_tuning_options,
                                                    "leftLine": patch_area_description,
                                                    "rightLine": patch_area_description,
                                                },
                                            },
                                        },
                                        {
                                            "if": {
                                                **crop_on_markers_if_requirements,
                                                "properties": {
                                                    "type": {"const": "FOUR_DOTS"}
                                                },
                                            },
                                            "then": {
                                                "required": [
                                                    "topRightDot",
                                                    "bottomRightDot",
                                                    "topLeftDot",
                                                    "bottomLeftDot",
                                                ],
                                                "additionalProperties": False,
                                                "properties": {
                                                    **crop_on_markers_options_available_keys,
                                                    "tuningOptions": crop_on_dot_lines_tuning_options,
                                                    "topRightDot": patch_area_description,
                                                    "bottomRightDot": patch_area_description,
                                                    "topLeftDot": patch_area_description,
                                                    "bottomLeftDot": patch_area_description,
                                                },
                                            },
                                        },
                                    ],
                                }
                            }
                        },
                    },
                ],
            },
        },
        "fieldBlocks": {
            "description": "The fieldBlocks denote small groups of adjacent fields",
            "type": "object",
            "patternProperties": {
                "^.*$": {
                    "type": "object",
                    "required": [
                        "origin",
                        "bubblesGap",
                        "labelsGap",
                        "fieldLabels",
                    ],
                    "oneOf": [
                        {"required": ["fieldType"]},
                        {"required": ["bubbleValues", "direction"]},
                    ],
                    "properties": {
                        "bubbleDimensions": two_positive_numbers,
                        "bubblesGap": positive_number,
                        "bubbleValues": ARRAY_OF_STRINGS,
                        "direction": {
                            "type": "string",
                            "enum": ["horizontal", "vertical"],
                        },
                        "emptyValue": {"type": "string"},
                        "fieldLabels": {"type": "array", "items": FIELD_STRING_TYPE},
                        "labelsGap": positive_number,
                        "origin": two_positive_integers,
                        "fieldType": {
                            "type": "string",
                            "enum": list(FIELD_TYPES.keys()),
                        },
                    },
                }
            },
        },
        "emptyValue": {
            "description": "The value to be used in case of empty bubble detected at global level.",
            "type": "string",
        },
    },
}

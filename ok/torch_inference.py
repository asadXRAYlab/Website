"""Anomalib Torch Inferencer Script.

This script performs torch inference by reading model weights
from command line, and show the visualization results.
"""

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import warnings
from argparse import ArgumentParser, Namespace
from pathlib import Path
from django.http import JsonResponse

import torch

from anomalib.data.utils import generate_output_image_filename, get_image_filenames, read_image
from anomalib.deploy import TorchInferencer
from anomalib.post_processing import Visualizer



def infer(args: Namespace) -> None:
    try:
        """Infer predictions.

        Show/save the output if path is to an image. If the path is a directory, go over each image in the directory.

        Args:
            args (Namespace): The arguments from the command line.
        """

        torch.set_grad_enabled(False)
        # Create the inferencer and visualizer.
        inferencer = TorchInferencer(path=args.weights, device=args.device)
        visualizer = Visualizer(mode=args.visualization_mode, task=args.task)

        filenames = get_image_filenames(path=args.input)
        for filename in filenames:
            image = read_image(filename)
            predictions = inferencer.predict(image=image)
            output = visualizer.visualize_image(predictions)
        
            if args.output is None and args.show is False:
                warnings.warn(
                    "Neither output path is provided nor show flag is set. Inferencer will run but return nothing."
                )

            if args.output:
                file_path = generate_output_image_filename(input_path=filename, output_path=args.output)
                visualizer.save(file_path=file_path, image=output)
                return output

             
            if args.show:
                visualizer.show(title="Output Image", image=output)
    except Exception as e:
            error_message = str(e)
            return JsonResponse({'error': error_message}, status=500)



# coding=utf-8
# Copyright 2024 Microsoft and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Processor class for Florence-2.
"""

import re
import logging
from typing import List, Optional, Union
import numpy as np

import torch
import PIL

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    PaddingStrategy,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType
import re

logger = logging.getLogger(__name__)


class Florence2Processor(ProcessorMixin):
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "CLIPImageProcessor"
    tokenizer_class = ("BartTokenizer", "BartTokenizerFast")

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
    ):
        if image_processor is None:
            raise ValueError("You need to specify an `image_processor`.")
        if tokenizer is None:
            raise ValueError("You need to specify a `tokenizer`.")
        if not hasattr(image_processor, "image_seq_length"):
            raise ValueError("Image processor is missing an `image_seq_length` attribute.")

        self.image_seq_length = image_processor.image_seq_length

        tokens_to_add = {
                'additional_special_tokens': \
                    tokenizer.additional_special_tokens + \
                    ['<od>', '</od>', '<ocr>', '</ocr>'] + \
                    [f'<loc_{x}>' for x in range(1000)] + \
                    ['<cap>', '</cap>', '<ncap>', '</ncap>','<dcap>', '</dcap>', '<grounding>', '</grounding>', '<seg>', '</seg>', '<sep>', '<region_cap>', '</region_cap>', '<region_to_desciption>', '</region_to_desciption>', '<proposal>', '</proposal>', '<poly>', '</poly>', '<and>'] + \
                    ['<panel>', '<text>', '<character>', '<tail>']
            }
        tokenizer.add_special_tokens(tokens_to_add)
        self.decoder_start_token_id = 2

        self.box_quantizer = BoxQuantizer(
            mode='floor',
            bins=(1000, 1000),
        )

        super().__init__(image_processor, tokenizer)
    
    def __call__(
        self,
        batch_input_text: List[TextInput] = None,
        batch_input_list_of_list_of_bboxes: List[List[List[List[float]]]] = None,
        batch_output_text: List[TextInput] = None,
        batch_output_list_of_list_of_bboxes: List[List[List[List[float]]]] = None,
        batch_images: ImageInput = None,
        batch_character_cluster_labels = None,
        batch_text_character_association_labels = None,
        batch_text_tail_association_labels = None,
        batch_is_essential_text_labels = None,
        batch_tail_character_association_labels = None,
        padding: Union[bool, str, PaddingStrategy] = None,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_input_length_including_image_tokens=None,
        max_output_length=None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
        do_resize: bool = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        data_format: Optional["ChannelDimension"] = "channels_first",  # noqa: F821
        input_data_format: Optional[
            Union[str, "ChannelDimension"]  # noqa: F821
        ] = None,
        resample: "PILImageResampling" = None,  # noqa: F821
        do_convert_rgb: bool = None,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ) -> BatchFeature:

        assert batch_images is not None, "`batch_images` are expected as arguments to a `Florence2Processor` instance."
        assert batch_input_text is not None, "`batch_input_text` are expected as arguments to a `Florence2Processor` instance."
        if batch_input_list_of_list_of_bboxes is None:
            batch_input_list_of_list_of_bboxes = [[] for _ in range(len(batch_input_text))]
        assert len(batch_input_text) == len(batch_input_list_of_list_of_bboxes) == len(batch_images), "`batch_input_text`, `batch_input_list_of_list_of_bboxes` and `batch_images` have different lengths."
        if batch_output_text is None:
            assert batch_output_list_of_list_of_bboxes is None, "`batch_output_text` and `batch_output_list_of_list_of_bboxes` should be provided together."
        else:
            if batch_output_list_of_list_of_bboxes is None:
                batch_output_list_of_list_of_bboxes = [[] for _ in range(len(batch_output_text))]
            assert len(batch_output_text) == len(batch_output_list_of_list_of_bboxes) == len(batch_images), "`batch_output_text`, `batch_output_list_of_list_of_bboxes` and `batch_images` have different lengths."

        max_input_length = max_input_length_including_image_tokens - self.image_seq_length if max_input_length_including_image_tokens is not None else None
        batch_input_texts = [self._format_text_with_bboxes(text, list_of_list_of_bboxes, image) for text, list_of_list_of_bboxes, image in zip(batch_input_text, batch_input_list_of_list_of_bboxes, batch_images)]
        inputs = self.tokenizer(
            batch_input_texts,
            return_tensors=return_tensors,
            padding=padding,
            truncation=False,
        )
        # Truncating manually because I don't want </s> token at the end of truncated sequences, which is the default behavior
        if inputs["input_ids"].shape[1] > max_input_length:
            inputs["input_ids"] = inputs["input_ids"][:, :max_input_length]
            inputs["attention_mask"] = inputs["attention_mask"][:, :max_input_length]
            
        if batch_output_text is not None:
            batch_output_texts = [self._format_text_with_bboxes(text, list_of_list_of_bboxes, image) for text, list_of_list_of_bboxes, image in zip(batch_output_text, batch_output_list_of_list_of_bboxes, batch_images)]
            decoder_inputs = self.tokenizer(
                batch_output_texts,
                return_tensors=return_tensors,
                padding=padding,
                truncation=False,
            )
            # Truncating manually because I don't want </s> token at the end of truncated sequences, which is the default behavior
            if decoder_inputs["input_ids"].shape[1] > max_output_length:
                decoder_inputs["input_ids"] = decoder_inputs["input_ids"][:, :max_output_length]
                decoder_inputs["attention_mask"] = decoder_inputs["attention_mask"][:, :max_output_length]
        

        pixel_values = self.image_processor(
            batch_images,
            do_resize=do_resize,
            do_normalize=do_normalize,
            return_tensors=return_tensors,
            image_mean=image_mean,
            image_std=image_std,
            input_data_format=input_data_format,
            data_format=data_format,
            resample=resample,
            do_convert_rgb=do_convert_rgb,
        )["pixel_values"]

        if dtype is not None:
            pixel_values = pixel_values.to(dtype)
        
        return_data = {**inputs, "pixel_values": pixel_values}

        if batch_output_text is not None:
            labels = decoder_inputs["input_ids"]
            decoder_input_ids = labels.new_zeros(labels.shape)
            decoder_input_ids[:, 1:] = labels[:, :-1].clone()
            decoder_input_ids[:, 0] = self.decoder_start_token_id
            decoder_attention_mask = decoder_inputs["attention_mask"].new_ones(decoder_input_ids.shape)
            decoder_attention_mask[:, 1:] = decoder_inputs["attention_mask"][:, :-1].clone()
            # Mask fill labels to replace pad token ID with -100
            labels.masked_fill_(labels == self.tokenizer.pad_token_id, -100)
            return_data.update({
                "labels": labels,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask": decoder_attention_mask,
            })
        
        if device is not None:
            for key, value in return_data.items():
                if isinstance(value, torch.Tensor):
                    return_data[key] = value.to(device)

        if batch_character_cluster_labels is not None:
            return_data["character_cluster_labels"] = batch_character_cluster_labels
        if batch_text_character_association_labels is not None:
            return_data["text_character_association_labels"] = batch_text_character_association_labels
        if batch_text_tail_association_labels is not None:
            return_data["text_tail_association_labels"] = batch_text_tail_association_labels
        if batch_is_essential_text_labels is not None:
            return_data["is_essential_text_labels"] = batch_is_essential_text_labels
        if batch_tail_character_association_labels is not None:
            return_data["tail_character_association_labels"] = batch_tail_character_association_labels

        return_data["tokenizer"] = self.tokenizer
        return BatchFeature(data=return_data)

    def cleanup_generated_text(self, generated_text):
        return generated_text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")

    def postprocess_output(self, generated_ids, images):
        generated_ids.masked_fill_(generated_ids == -100, self.tokenizer.pad_token_id) # only for some testing purposes
        batch_decoded_texts = self.batch_decode(generated_ids, skip_special_tokens=False)
        batch_decoded_texts = [self.cleanup_generated_text(text) for text in batch_decoded_texts]
        batch_list_of_list_of_bboxes = []
        batch_indices_of_bboxes_in_new_string = []
        batch_new_texts = []
        for text, image in zip(batch_decoded_texts, images):
            size_wh = self._get_image_size_wh(image)
            parsed_text, list_of_stringified_bboxes, start_end_in_new_string = self._parse_text_with_bboxes(text)
            list_of_list_of_bboxes = [self.box_quantizer.dequantize_from_stringified_bboxes(stringified_bbox, size_wh) for stringified_bbox in list_of_stringified_bboxes]
            batch_list_of_list_of_bboxes.append(list_of_list_of_bboxes)
            batch_indices_of_bboxes_in_new_string.append(start_end_in_new_string)
            batch_new_texts.append(parsed_text)
        return batch_new_texts, batch_list_of_list_of_bboxes, batch_indices_of_bboxes_in_new_string

    def _parse_text_with_bboxes(self, text):
        loc_pattern = r'((?:<loc_\d+>){4}(?:,(?:<loc_\d+>){4})*)'
        grounding_pattern = r'<grounding>(.*?)</grounding>' + loc_pattern
        
        list_of_stringified_bboxes = []
        start_end_in_new_string = []
        new_text = ""
        original_pos = 0
        new_pos = 0

        for match in re.finditer(grounding_pattern + '|' + loc_pattern, text):
            # Add text before the match
            new_text += text[original_pos:match.start()]
            new_pos += match.start() - original_pos

            if match.group(0).startswith('<grounding>'):
                # Handle grounding pattern
                grounding_text = match.group(1)
                locs = match.group(2)
                new_text += grounding_text
                list_of_stringified_bboxes.append(locs)
                start_end_in_new_string.append((new_pos, new_pos + len(grounding_text)))
                new_pos += len(grounding_text)
            else:
                # Handle loc pattern
                locs = match.group(0)
                replacement = ""
                new_text += replacement
                list_of_stringified_bboxes.append(locs)
                start_end_in_new_string.append((new_pos, new_pos + len(replacement)))
                new_pos += len(replacement)

            original_pos = match.end()

        # Add any remaining text
        new_text += text[original_pos:]

        return new_text, list_of_stringified_bboxes, start_end_in_new_string
    
    def _format_text_with_bboxes(self, text, list_of_list_of_bboxes, image):
        size_wh = self._get_image_size_wh(image)
        quantized_bbox_lists = []
        for list_of_bboxes in list_of_list_of_bboxes:            
            quantized_bboxes = self.box_quantizer.quantize(list_of_bboxes, size_wh=size_wh)
            stringified_bboxes = [f"<loc_{x1}><loc_{y1}><loc_{x2}><loc_{y2}>" for x1, y1, x2, y2 in quantized_bboxes]
            stringified_bboxes = ",".join(stringified_bboxes)
            quantized_bbox_lists.append(stringified_bboxes)
        return text.format(*quantized_bbox_lists)

    def _get_image_size_wh(self, image):
         # Get size_wh from image based on its type
        if isinstance(image, torch.Tensor):
            # For PyTorch tensor
            if image.dim() == 3:
                size_wh = (image.shape[2], image.shape[1])  # (width, height)
            elif image.dim() == 4:
                size_wh = (image.shape[3], image.shape[2])  # (width, height)
            else:
                raise ValueError("Unsupported tensor dimensions")
        elif isinstance(image, np.ndarray):
            # For NumPy array
            if image.ndim == 2:
                size_wh = (image.shape[1], image.shape[0])  # (width, height)
            elif image.ndim == 3:
                size_wh = (image.shape[1], image.shape[0])  # (width, height)
            else:
                raise ValueError("Unsupported array dimensions")
        elif isinstance(image, PIL.Image.Image):
            # For PIL Image
            size_wh = image.size  # Already in (width, height) format
        else:
            raise TypeError("Unsupported image type")
        return size_wh

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.batch_decode with CLIP->Florence2
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BartTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->Florence2
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BartTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.model_input_names with CLIP->Florence2
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

class BoxQuantizer(object):
    def __init__(self, mode, bins):
        self.mode = mode
        self.bins = bins

    def quantize(self, boxes, size_wh):
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes)
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size_wh       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            quantized_xmin = (
                xmin / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymin = (
                ymin / size_per_bin_h).floor().clamp(0, bins_h - 1)
            quantized_xmax = (
                xmax / size_per_bin_w).floor().clamp(0, bins_w - 1)
            quantized_ymax = (
                ymax / size_per_bin_h).floor().clamp(0, bins_h - 1)

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        quantized_boxes = torch.cat(
            (quantized_xmin, quantized_ymin, quantized_xmax, quantized_ymax), dim=-1
        ).int()

        return quantized_boxes.tolist()

    def dequantize_from_stringified_bboxes(self, stringified_bboxes, size_wh):
        bboxes = stringified_bboxes.split(',')

        def parse_bbox(bbox_string):
            pattern = r'<loc_(\d+)><loc_(\d+)><loc_(\d+)><loc_(\d+)>'
            match = re.match(pattern, bbox_string)
            if match:
                return [int(match.group(i)) for i in range(1, 5)]
            else:
                raise ValueError(f"Invalid bbox string format: {bbox_string}")

        parsed_bboxes = [parse_bbox(bbox) for bbox in bboxes]
        return self.dequantize(parsed_bboxes, size_wh).tolist()

    def dequantize(self, boxes: torch.Tensor, size):
        if not isinstance(boxes, torch.Tensor):
            boxes = torch.tensor(boxes)
        bins_w, bins_h = self.bins  # Quantization bins.
        size_w, size_h = size       # Original image size.
        size_per_bin_w = size_w / bins_w
        size_per_bin_h = size_h / bins_h
        xmin, ymin, xmax, ymax = boxes.split(1, dim=-1)  # Shape: 4 * [N, 1].

        if self.mode == 'floor':
            # Add 0.5 to use the center position of the bin as the coordinate.
            dequantized_xmin = (xmin + 0.5) * size_per_bin_w
            dequantized_ymin = (ymin + 0.5) * size_per_bin_h
            dequantized_xmax = (xmax + 0.5) * size_per_bin_w
            dequantized_ymax = (ymax + 0.5) * size_per_bin_h

        elif self.mode == 'round':
            raise NotImplementedError()

        else:
            raise ValueError('Incorrect quantization type.')

        dequantized_boxes = torch.cat(
            (dequantized_xmin, dequantized_ymin,
             dequantized_xmax, dequantized_ymax), dim=-1
        )

        return dequantized_boxes
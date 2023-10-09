# -*- coding:utf-8 -*-
# create: @time: 10/8/23 11:47
import argparse

import torch
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from nougat_latex.util import process_raw_latex_code

from nougat_latex import NougatLaTexProcessor
from nougat_latex.image_processing_nougat import NougatImageProcessor


def parse_option():
    parser = argparse.ArgumentParser(prog="nougat inference config", description="model archiver")
    parser.add_argument("--pretrained_model_name_or_path", default="Norm/nougat-latex-base")
    parser.add_argument("--img_path", help="path to latex image segment", required=True)
    parser.add_argument("--device", default="gpu")
    return parser.parse_args()


def run_nougat_latex():
    args = parse_option()
    # device
    if args.device == "gpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    # init model
    model = VisionEncoderDecoderModel.from_pretrained(args.pretrained_model_name_or_path).to(device)

    # init processor
    tokenizer = NougatTokenizerFast.from_pretrained(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path
    )

    image_processor = NougatImageProcessor.from_pretrained(args.pretrained_model_name_or_path)
    latex_processor = NougatLaTexProcessor(image_processor=image_processor)

    # run test
    image = Image.open(args.img_path)
    if not image.mode == "RGB":
        image = image.convert('RGB')

    pixel_values = latex_processor(image)
    task_prompt = tokenizer.bos_token
    decoder_input_ids = tokenizer(task_prompt, add_special_tokens=False,
                                  return_tensors="pt").input_ids
    with torch.no_grad():
        outputs = model.generate(
            pixel_values.to(device),
            decoder_input_ids=decoder_input_ids.to(device),
            max_length=model.decoder.config.max_length,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    sequence = tokenizer.batch_decode(outputs.sequences)[0]
    sequence = sequence.replace(tokenizer.eos_token, "").replace(tokenizer.pad_token, "").replace(tokenizer.bos_token, "")
    sequence = process_raw_latex_code(sequence)
    print(sequence)


if __name__ == '__main__':
    run_nougat_latex()

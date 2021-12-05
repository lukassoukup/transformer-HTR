import argparse
import os
from pathlib import Path
import torch

from model import TransformerModel
from utils import vocab_size, tokens, load_image
from helpers.save_outputs import write_predict


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer parameters', add_help=False)

    parser.add_argument('--input_dir', required=True, type=str, help='Directory with input images')
    parser.add_argument('--model_file', required=True, type=str, help='Path to trained checkpoint')
    parser.add_argument('--output_dir', default='output/',
                        help="Path for saving the outputs")
    parser.add_argument('--device', default="cuda", help="Device to be used for inference.")

    return parser


def predict(model, input_sequence, device, max_length=1000):  #(OUTPUT_MAX_LEN - 1)):
    """
    Method from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """
    model.eval()

    y_input = torch.tensor([[tokens['GO_TOKEN']]], dtype=torch.long, device=device)

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1)).to(device)

        pred = model(input_sequence, y_input, tgt_mask=tgt_mask)

        next_item = pred.topk(1)[1].view(-1)[-1].item()  # num with highest probability
        next_item = torch.tensor([[next_item]], device=device)

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == tokens['END_TOKEN']:
            break

    return y_input.view(-1).tolist()


def make_prediction(model, input_dir, device):
    device = torch.device(device)
    sequence_max_length = 35  # the longest sequence in the training data

    for img_name in input_dir.iterdir():
        img, img_width = load_image(img_name)

        source = torch.from_numpy(img).to(device)

        result = predict(model, source, device, max_length=sequence_max_length)
        write_predict(0, img_name.name, result[1:-1], 'test', args)

        print(f"Example {img_name.name}")
        print(f"Prediction: {result}")
        print()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser('Transformer training and evaluation script', parents=[get_args_parser()])
    args = arg_parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    model_file = args.model_file
    print('Loading ' + model_file)
    checkpoint = torch.load(model_file, map_location=device)

    patch_width = checkpoint['patch_width']
    model_dim = checkpoint['model_dim']
    num_heads = checkpoint['num_heads']
    num_enc_layers = checkpoint['num_enc_layers']
    num_dec_layers = checkpoint['num_dec_layers']

    model = TransformerModel(vocab_size, model_dim, num_heads, num_enc_layers, num_dec_layers, patch_width).to(device)
    model.load_state_dict({ key.replace("module.", "") : value for key, value in checkpoint['model_state_dict'].items()})


    input_dir = Path(args.input_dir)

    make_prediction(model, input_dir, device)

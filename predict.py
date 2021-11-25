import argparse
import os
from pathlib import Path
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import TransformerModel
from utils import vocab_size, tokens, load_image
from helpers.save_outputs import write_predict


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer parameters', add_help=False)

    parser.add_argument('--input_dir', required=True, type=str, help='Directory with input images')
    parser.add_argument('--model_file', required=True, type=str, help='Path to trained checkpoint')
    parser.add_argument('--output_dir', default='output/',
                        help="Path for saving the outputs")
    parser.add_argument('--device', default="cuda", help="Device to be used for inference")

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
        tgt_mask = model.module.get_tgt_mask(y_input.size(1)).to(device)

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

        source = torch.from_numpy(img)
        source.to(device)

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

    try:
        gpus_per_node = int(os.environ['PBS_NGPUS'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        node_id = int(os.environ['OMPI_COMM_WORLD_RANK'])
        gpu_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        print("Successfully loaded parameters from environments")
    except KeyError:
        gpus_per_node = 1
        world_size = 1
        node_id = 0
        gpu_rank = 0
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '8888'
        print("KeyError, using default values")

    dist.init_process_group('nccl', rank=gpu_rank, world_size=world_size)

    model_file = args.model_file
    print('Loading ' + model_file)
    checkpoint = torch.load(model_file)

    patch_width = checkpoint['patch_width']
    model_dim = checkpoint['model_dim']
    num_heads = checkpoint['num_heads']
    num_enc_layers = checkpoint['num_enc_layers']
    num_dec_layers = checkpoint['num_dec_layers']

    device = torch.device(args.device)
    model = DDP(TransformerModel(vocab_size, model_dim, num_heads, num_enc_layers, num_dec_layers, patch_width)
                .to(device), device_ids=[0])
    model.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device(args.device)
    model.to(device)

    input_dir = Path(args.input_dir)

    make_prediction(model, input_dir, args.device)

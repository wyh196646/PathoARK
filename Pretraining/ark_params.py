import argparse


def get_ark_pretrain_params():
    parser = argparse.ArgumentParser(description="ARK-style supervised pretraining for pathology")

    # data/config
    parser.add_argument('--datasets_yaml', type=str, default='./datasets/ark_datasets_template.yaml', help='YAML listing datasets with paths/splits/num_classes')
    parser.add_argument('--save_dir', type=str, default='./outputs/ark_pretrain', help='Checkpoint directory')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32, help='Batches are slides; 1 is typical')

    # model
    parser.add_argument('--model_arch', type=str, default='gigapath_slide_enc12l768d')
    parser.add_argument('--input_dim', type=int, default=1536)
    parser.add_argument('--latent_dim', type=int, default=768)
    parser.add_argument('--feat_layer', type=str, default='11')
    parser.add_argument('--pretrained', type=str, default='')
    parser.add_argument('--freeze', action='store_true', default=False)
    parser.add_argument('--global_pool', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)

    # projector + EMA consistency
    parser.add_argument('--projector_features', type=int, default=768)
    parser.add_argument('--use_mlp', action='store_true', default=False)
    parser.add_argument('--ema_mode', type=str, default='epoch', choices=['epoch', 'iteration'])
    parser.add_argument('--momentum_teacher', type=float, default=0.9)

    # train
    parser.add_argument('--device', type=str, default='cuda', help='Device. In DDP 会被自动改成 cuda:local_rank')
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--optim_wd', type=float, default=1e-5)
    parser.add_argument('--test_epoch', type=int, default=1)

    # distributed
    parser.add_argument('--distributed', action='store_true', default=False, help='Enable DistributedDataParallel')
    parser.add_argument('--dist_url', type=str, default='env://', help='URL used to set up distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='Distributed backend')
    parser.add_argument('--world_size', type=int, default=-1, help='Number of processes (set automatically by torchrun)')
    parser.add_argument('--rank', type=int, default=-1, help='Rank of this process')
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed by torchrun')
    parser.add_argument('--find_unused_parameters', action='store_true', default=True,
                        help='Set torch.nn.parallel.DistributedDataParallel(find_unused_parameters=True). Required when not all heads are used every iteration.')

    return parser.parse_args()

exp_name = 'SpecGAN_16x_sr'

scale = 16
# model settings
model = dict(
    type='GLEAN',
    generator=dict(
        type='SpecGAN',
        in_size=16,
        out_size=256,
        img_in_channels=51,
        style_channels=512,
    ),
    discriminator=dict(
        type='StyleGAN2Discriminator',
        in_size=256,
    ),
    pixel_loss=dict(type='MSELoss', loss_weight=1.0, reduction='mean'),
    perceptual_loss=dict(
        type='PerceptualLoss',
        layer_weights={'21': 1.0},
        vgg_type='vgg16',
        perceptual_weight=1e-3,
        style_weight=0,
        norm_img=False,
        criterion='mse',
        pretrained='torchvision://vgg16'),
    gan_loss=dict(
        type='GANLoss',
        gan_type='vanilla',
        loss_weight=0.1,
        real_label_val=1.0,
        fake_label_val=0),
    pretrained=None,
)

# model training and testing settings
train_cfg = dict(disc_init_steps=0, disc_steps=1)
test_cfg = dict(metrics=['PSNR'], crop_border=0)

# dataset settings
train_pipeline = [
    dict(type='LoadTIFImageFromFile', io_backend='disk', key='lq'),
    dict(type='LoadTIFImageFromFile', io_backend='disk', key='gt'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq'],
        mean=0.5,
        std=0.5,
        to_rgb=False),
    dict(
        type='Normalize',
        keys=['gt'],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        to_rgb=False),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='vertical'),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]
test_pipeline = [
    dict(type='LoadTIFImageFromFile', io_backend='disk', key='lq'),
    dict(type='LoadTIFImageFromFile', io_backend='disk', key='gt'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(
        type='Normalize',
        keys=['lq'],
        mean=0.5,
        std=0.5,
        to_rgb=False),
    dict(
        type='Normalize',
        keys=['gt'],
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'lq_path'])
]

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(samples_per_gpu=4, drop_last=True),
    val_dataloader=dict(samples_per_gpu=1),
    test_dataloader=dict(samples_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='SRFolderDataset',
            lq_folder='F:/mengyapeng/dataset/grss2018-me/x16/train/HSI_cut',
            gt_folder='F:/mengyapeng/dataset/grss2018-me/x16/train/RGBHR_cut',
            pipeline=train_pipeline,
            scale=scale)),
    val=dict(
        type='SRFolderDataset',
        lq_folder='F:/mengyapeng/dataset/grss2018-me/x16/valid/HSI_cut',
        gt_folder='F:/mengyapeng/dataset/grss2018-me/x16/valid/RGBHR_cut',
        pipeline=test_pipeline,
        scale=scale),
    test=dict(
        type='SRFolderDataset',
        lq_folder='F:/mengyapeng/dataset/grss2018-me/x16/test/HSI_cut',
        gt_folder='F:/mengyapeng/dataset/grss2018-me/x16/test/RGBHR_cut',
        pipeline=test_pipeline,
        scale=scale))

# optimizer
optimizers = dict(
    generator=dict(type='Adam', lr=1e-5, betas=(0.9, 0.99)),
    discriminator=dict(type='Adam', lr=1e-5, betas=(0.9, 0.99)))

# learning policy
total_iters = 500000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[total_iters],
    restart_weights=[1],
    min_lr=1e-8)

checkpoint_config = dict(interval=10000, save_optimizer=False, by_epoch=False)
evaluation = dict(interval=10000, save_image=True)
log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./work_dirs/{exp_name}'
load_from = None
resume_from = ''  # pre-trained weight path
workflow = [('train', 1)]
find_unused_parameters = True

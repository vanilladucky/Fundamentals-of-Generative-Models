#!/ext4/pyenv/diffusers_pt2n/bin/python


#!/ext4/pyenv/diffusers/bin/python
#!/ext4/pyenv/diffusers_pt2n/bin/python
# !/home/sf/data/linux/pyenv/pt2/bin/python
import torch

#from tqdm import tqdm
from tqdm.auto import tqdm
import time, os, json
import numpy as np
from pprint import pprint
from shutil import copy2

from train_util import *
from artbench import set_dataloader_vq_imgfld, set_dataloader_disc_imgfld
#from src.datasets.huggan import set_dataloader_vq_hf, set_dataloader_disc_hf
#from src.datasets.laion_art import set_dataloader_vq_laion_art, set_dataloader_disc_laion_art
from disc_loss import DiscLoss
from train_util import init_discriminator
from KL_LPIPSWithDisc import KL_LPIPSWithDiscriminator
from utils_aux import unscale_tensor, save_grid_imgs, weights_init

# to avoid IO errors with massive reading of the files
import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')


def l2_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def conv2tqdm_msg(msg, fmt='>.5f'):
    res = {}
    for key in msg:
        if key != 'Step':
            res[key] = f'{msg[key]:{fmt}}'
        else:
            res[key] = f'{msg[key]}'
    return res


# ===================================================================
def main(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)
    print('\nThis config will be used\n')
    pprint(config)

    # ----------------------------------------------
    # Preconfig
    # ----------------------------------------------
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'

    seed = 19880122
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.preferred_linalg_library(backend='default')
    if config['training']['fp16'] == 'fp16':
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    else:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('medium')

    # Torch home folder for caching
    torch.hub.set_dir(os.path.join(os.getcwd(),'torch_hub_models'))
    os.environ['TORCH_HOME'] = os.path.join(os.getcwd(),'torch_hub_models')


    # for varying input sizes
    if config['training']['compile']:
        #torch._dynamo.config.dynamic_shapes = True
        torch._dynamo.config.assume_static_by_default = True
        torch._dynamo.config.automatic_dynamic_shapes = True
        torch._dynamo.config.suppress_errors = True


    # torch._dynamo.config.verbose=True

    # model name
    model_name = config['model']['name']

    # results and chkpts folders:
    cwd = config['results']['root']
    if cwd == 'cwd':
        cwd = os.getcwd()
    results_folder = os.path.join(cwd, model_name)
    chkpts = os.path.join(cwd, *(model_name, 'chkpts'))
    img_folder = os.path.join(cwd, *(model_name, 'samples'))
    os.makedirs(chkpts, exist_ok=True)
    os.makedirs(img_folder, exist_ok=True)

    # copy the config
    _f_dst = os.path.join(results_folder, 'train_config.json')
    copy2(config_file, _f_dst)

    # Dataloader
    if config['dataset'].get('type', 'legacy') == 'legacy':
        train_loader = set_dataloader_vq_imgfld(config)
        if config['discriminator']['disc_train_batch']:
            disc_loader = set_dataloader_disc_imgfld(config)
        else:
            disc_loader = train_loader
    """elif config['dataset'].get('type', 'legacy') == 'hf':
        train_loader = set_dataloader_vq_hf(config)
        if config['discriminator']['disc_train_batch']:
            disc_loader = set_dataloader_disc_hf(config)
        else:
            disc_loader = train_loader
    elif config['dataset'].get('type', 'legacy').lower() == 'laion_art':
        train_loader = set_dataloader_vq_laion_art(config)
        if config['discriminator']['disc_train_batch']:
            disc_loader = set_dataloader_disc_laion_art(config)
        else:
            disc_loader = train_loader"""
    # number of batches in the train loader
    # in case of WebDataset, len is not defined
    # in this case check "loader_max_len" in "training"
    # !!! rollback value is set to 1e5 just because we need some !!!
    try:
        max_len = len(train_loader)
    except:
        max_len = config['training'].get('loader_max_len', 1e5)

    # Train parameters
    num_epochs = int(config['training']['num_epochs'])
    grad_step_acc = int(config['training']['grad_step_acc'])
    sample_every = config['training']['sample_every']
    if sample_every:
        sample_every = int(sample_every)
    save_every = int(config['training']['save_every'])
    save_snapshot = config['training']['save_snapshot']
    flag_compile = config['training']['compile']
    if save_snapshot:
        print(f'Model snapshots will be saved in {chkpts}')
    last_layer = config['training'].get('last_layer', True)

    # mixed precision
    if 'bfloat' in config['training']['fp16'] or 'bf16' in config['training']['fp16']:
        fp16 = torch.bfloat16
    if config['training']['fp16'] == 'fp16':
        fp16 = torch.float16
    if not config['training']['fp16']:
        fp16 = None

    # device
    device_model = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", torch.cuda.get_device_name(device_model))

    # init the model
    model, model_eager = prepare_vaemodel(config, device_model, flag_compile)
    if config['training']['fp16'] == 'fp16':
        model = model.apply(weights_init)
        model_eager = model_eager.apply(weights_init)
    encoder_tanh = config['model'].get('encode_tanh_out', False)
    # init the disc
    if config['discriminator']['enabled']:
        discriminator = init_discriminator(config['discriminator'], device_model)
    else:
        discriminator = None

    # EMA
    ema_flag = False
    if 'ema' in config:
        ema_flag = config['ema'].get('enabled', False)
    if ema_flag:
        ema_model = set_ema_model(model_eager, config['ema'])
    else:
        ema_model = None

    print('Setting optimizer, scheduler, and scaler')
    # set the model optimizers
    m_opt = set_optimizer(config['optimizer'], model, flag_compile)
    if discriminator:
        d_opt = set_optimizer(config['optimizer_d'], discriminator, flag_compile)
    else:
        d_opt = None

    # Schedulers
    if 'type' not in config['lr_scheduler']:
        raise ValueError
    if config['lr_scheduler']['type'].lower() == 'onecyclelr':
        config['lr_scheduler']['epochs'] = num_epochs
        if config['lr_scheduler']['steps_per_epoch'] is None:
            config['lr_scheduler']['steps_per_epoch'] = int(
                np.ceil(max_len / config['training']['grad_step_acc']))
    m_sch = set_lr_scheduler(config['lr_scheduler'], m_opt)
    if discriminator:
        if config['lr_scheduler_d']['type'].lower() == 'onecyclelr':
            config['lr_scheduler_d']['epochs'] = num_epochs
            if config['lr_scheduler_d']['steps_per_epoch'] is None:
                config['lr_scheduler_d']['steps_per_epoch'] = int(
                    np.ceil(max_len / config['training']['grad_step_acc']))
        d_sch = set_lr_scheduler(config['lr_scheduler_d'], d_opt)
    else:
        d_sch = None

    # Scaler
    if config['training']['fp16']:
        init_scale = config['training'].get('init_scale', 16384)
        growth_interval = config['training'].get('growth_interval', 10)
        growth_factor = config['training'].get('growth_factor', 2)
        backoff_factor = config['training'].get('backoff_factor', 0.5)
        m_scaler = torch.cuda.amp.GradScaler(init_scale=init_scale,
                                             growth_factor=growth_factor,
                                             growth_interval=growth_interval,
                                             backoff_factor=backoff_factor)
        d_scaler = torch.cuda.amp.GradScaler(init_scale=init_scale,
                                             growth_factor=growth_factor,
                                             growth_interval=growth_interval,
                                             backoff_factor=backoff_factor)
    else:
        m_scaler = None
        d_scaler = None
    print('Done')

    # ----------------------------------------------
    # Discriminator pretrain
    # ----------------------------------------------
    if config['discriminator']['disc_pretrain_epochs']:
        disc_loss = DiscLoss(discriminator, 'hinge')
        d_grad_step_acc = config['discriminator']['disc_grad_acc']
        d_epochs = config['discriminator']['disc_pretrain_epochs']
        print(
            f'---------------------------------------------\nPretraining the discriminator for {d_epochs:>3} epochs\n---------------------------------------------')
        discriminator = pretrain_discriminator(discriminator, model, disc_loader,
                                               disc_loss, d_opt, d_sch,
                                               d_scaler, d_epochs, device_model, fp16,
                                               d_grad_step_acc)
        disc_name = os.path.join(chkpts, 'discriminator.pth')
        save_model(discriminator, disc_name)
        # re-init the optimizer, scaler, and scheduler
        d_opt = set_optimizer(config['optimizer_d'], discriminator, flag_compile)
        d_sch = set_lr_scheduler(config['lr_scheduler_d'], m_opt)
        d_scaler = torch.cuda.amp.GradScaler()
        print(f'Saved pretrained discriminator in\n\t{disc_name}')

    # ----------------------------------------------
    # Set LPIPS with Disc
    # ----------------------------------------------
    LPIPSDiscLoss = KL_LPIPSWithDiscriminator(discriminator=discriminator,
                                              cfg=config['loss']).to(device_model)

    # ----------------------------------------------
    # Save original images
    # ----------------------------------------------
    x_original = next(iter(train_loader))
    x_original = x_original[0]
    save_grid_imgs(unscale_tensor(x_original), max(x_original.shape[0] // 4, 2),
                   f'{results_folder}/original-images.jpg')
    x_original = x_original.to(device_model, non_blocking=True)
    # ----------------------------------------------
    # Training
    # ----------------------------------------------
    total_loss = []
    step = 0
    disc_name = os.path.join(chkpts, 'discriminator_upd.pth')

    # ---------------------------------------------------
    # Loading optimizer and scaler from a chkpt if exists
    # ---------------------------------------------------
    chkpt_path = config['training'].get('chkpt', None)
    epoch_start = 0
    if chkpt_path:
        print(f'Loading opt., scaler, sch. from {chkpt_path}')
        checkpoint = torch.load(chkpt_path)
        if 'm_opt' in checkpoint:
            m_opt.load_state_dict(checkpoint['m_opt'])
        if 'm_scaler' in checkpoint:
            if m_scaler:
                m_scaler.load_state_dict(checkpoint['m_scaler'])
                print(f'\tModel scaler: {m_scaler.get_scale()}')
        if 'd_opt' in checkpoint:
            d_opt.load_state_dict(checkpoint['d_opt'])
        if 'd_scaler' in checkpoint:
            if d_scaler:
                d_scaler.load_state_dict(checkpoint['d_scaler'])
                print(f'\tDiscriminator scaler: {m_scaler.get_scale()}')
        if 'm_sch' in checkpoint:
            if m_sch:
                m_sch.load_state_dict(checkpoint['m_sch'])
        if 'd_sch' in checkpoint:
            if d_sch:
                d_sch.load_state_dict(checkpoint['d_sch'])
        epoch_start = checkpoint.get('epoch', 0)
        step = checkpoint.get('step', 0)
        print('Done')
    # ---------------------------------------------------
    # Training
    # ---------------------------------------------------
    print(f'Training for {num_epochs - epoch_start} epochs')
    print(f'Sampling every {sample_every} steps')
    print(f'Saving every {save_every} steps')
    print(
        '==============================================================\n\t\tTraining\n==============================================================\n')
    m_loss_log = []
    d_loss_log = []
    m_loss_fname = os.path.join(results_folder, 'model_log_loss.csv')
    d_loss_fname = os.path.join(results_folder, 'disc_log_loss.csv')
    nll_w = config['loss'].get('nll_weight', 1)

    for epoch in range(epoch_start, num_epochs):
        t_start = time.time()
        progress_bar = tqdm(train_loader, desc=f'Train {epoch + 1}', total=max_len, position=0,
                            mininterval=1.0, leave=False, disable=False, colour='#009966')

        avg_tot = 0
        avg_metrics = {}

        d_opt.zero_grad(set_to_none=True)
        m_opt.zero_grad(set_to_none=True)

        for bstep, X in enumerate(progress_bar):
            batch = X[0].to(device_model, non_blocking=True)

            # Forward pass for VAE
            with torch.cuda.amp.autocast(dtype=fp16):
                batch_recon, posterior = model(batch)

            # GAN-like part
            with torch.cuda.amp.autocast(dtype=fp16):
                m_loss, m_msg = LPIPSDiscLoss(batch, batch_recon, posterior, 0, step,
                                              last_layer=model.get_last_layer() if last_layer else None,
                                              weights=nll_w)

            m_scaler.scale(m_loss / grad_step_acc).backward()

            m_msg_tqdm = conv2tqdm_msg(m_msg)
            progress_bar.set_postfix(m_msg_tqdm)
            # add to the avg metrics
            # and to the log
            t = []
            for key in m_msg:
                if key not in avg_metrics:
                    avg_metrics[key] = 0
                avg_metrics[key] += m_msg[key]
                t.append(m_msg[key])
            if m_sch:
                t.append(m_sch.get_last_lr()[0])
            m_loss_log.append(t)

            # Update the discriminator
            with torch.cuda.amp.autocast(dtype=fp16):
                d_loss, d_msg = LPIPSDiscLoss(batch, batch_recon, posterior, 1, step,
                                              last_layer=model.get_last_layer() if last_layer else None,
                                              weights=nll_w)

            d_scaler.scale(d_loss / grad_step_acc).backward()

            t = []
            for key in d_msg:
                t.append(d_msg[key])
            else:
                t.append(config['optimizer']['lr'])
            if d_sch:
                t.append(d_sch.get_last_lr()[0])
            else:
                t.append(config['optimizer_d']['lr'])
            d_loss_log.append(t)

            # update scaler, optimizer, and backpropagate
            if step != 0 and step % grad_step_acc == 0 or (bstep + 1) == max_len:
                # model part
                m_scaler.step(m_opt)
                m_scaler.update()
                if m_sch:
                    try:
                        m_sch.step()
                    except Exception as e:
                        print(e)
                m_opt.zero_grad(set_to_none=True)
                # discriminator part
                d_scaler.step(d_opt)
                d_scaler.update()
                if d_sch:
                    try:
                        d_sch.step()
                    except Exception as e:
                        print(e)
                d_opt.zero_grad(set_to_none=True)

                if ema_model:
                    ema_model.update()

            total_loss.append(m_loss.detach().to('cpu').item())
            avg_tot += total_loss[-1]

            # save generated images
            if sample_every:
                if step != 0 and (step + 1) % sample_every == 0:
                    with torch.no_grad():
                        x_recon, *_ = model(x_original)
                    x_recon = unscale_tensor(x_recon).detach()
                    save_grid_imgs(x_recon, max(x_recon.shape[0] // 4, 2), \
                                   f'{img_folder}/recon_imgs-{step + 1}-{epoch + 1}.jpg')
                    # model and discriminator snapshots
                    torch.save(model_eager.state_dict(), f'{chkpts}/sample_model_orig_snapshot.pt')
                    save_model(discriminator, disc_name)
                    del x_recon

                    if ema_model:
                        torch.save(ema_model.ema_model.state_dict(), f'{chkpts}/EMA_sample_snapshot.pt')
                        try:
                            with torch.no_grad():
                                x_recon, *_ = ema_model.ema_model(x_original)
                            x_recon = unscale_tensor(x_recon).detach()
                            save_grid_imgs(x_recon, max(x_recon.shape[0] // 4, 2), \
                                           f'{img_folder}/EMA_recon_imgs-{step + 1}-{epoch + 1}.jpg')
                            del x_recon
                        except Exception as E:
                            print(E)

                    # training snapshot
                    checkpoint = {
                        'epoch': epoch,
                        'step': step,
                        'm_opt': m_opt.state_dict(),
                        'm_scaler': m_scaler.state_dict() if m_scaler else None,
                        'd_opt': d_opt.state_dict(),
                        'd_scaler': d_scaler.state_dict() if d_opt else None,
                        'm_sch': m_sch.state_dict() if m_sch else None,
                        'd_sch': m_sch.state_dict() if d_sch else None,
                    }
                    torch.save(checkpoint, f'{chkpts}/sample_chkpt_snapshot.pt')

            # save checkpoints
            if step != 0 and (step + 1) % save_every == 0:
                checkpoint = {
                    'epoch': epoch,
                    'step': step,
                    'm_opt': m_opt.state_dict(),
                    'm_scaler': m_scaler.state_dict() if m_scaler else None,
                    'd_opt': d_opt.state_dict(),
                    'd_scaler': d_scaler.state_dict() if d_opt else None,
                    'm_sch': m_sch.state_dict() if m_sch else None,
                    'd_sch': m_sch.state_dict() if d_sch else None,
                }
                torch.save(checkpoint, f'{chkpts}/chkpt_snapshot_s{step + 1}-e{epoch + 1}.pt')
                torch.save(model_eager.state_dict(), f'{chkpts}/model_orig_s{step + 1}-e{epoch + 1}.pt')
                if ema_model:
                    torch.save(ema_model.ema_model.state_dict(), f'{chkpts}/EMA_model_orig.pt')
                save_model(discriminator, disc_name)

            # Update the global step
            step += 1

            # save the losses
            t = np.array(m_loss_log)
            np.savetxt(m_loss_fname, t, header=','.join(list(m_msg.keys()) + ['lr']), delimiter=",")
            t = np.array(d_loss_log)
            np.savetxt(d_loss_fname, t, header=','.join(list(d_msg.keys()) + ['lr']), delimiter=",")

        # save a snapshot after each epoch
        if save_snapshot:
            torch.save(model_eager.state_dict(), f'{chkpts}/model_orig_snapshot.pt')
            checkpoint = {
                'epoch': epoch,
                'step': step,
                'm_opt': m_opt.state_dict(),
                'm_scaler': m_scaler.state_dict() if m_scaler else None,
                'd_opt': d_opt.state_dict(),
                'd_scaler': d_scaler.state_dict() if d_opt else None,
                'm_sch': m_sch.state_dict() if m_sch else None,
                'd_sch': m_sch.state_dict() if d_sch else None,
            }
            torch.save(checkpoint, f'{chkpts}/chkpt_snapshot.pt')

        # save the discriminator after each epoch
        save_model(discriminator, disc_name)

        # print averaged data
        avg_tot /= (bstep + 1)
        print(f'Epoch {epoch + 1} in {time.time() - t_start:.2f}')
        try:
            msg = '\t---->'
            for key in avg_metrics:
                if key != 'Step':
                    msg += f' {key}: {avg_metrics[key] / (bstep + 1):>.5f};'
            print(msg)
        except Exception as e:
            print(e)

        # sample in the end of the training epoch
        with torch.no_grad():
            x_recon, *_ = model(x_original)
        x_recon = unscale_tensor(x_recon).detach()
        save_grid_imgs(x_recon, max(x_recon.shape[0] // 4, 2), \
                       f'{img_folder}/recon_imgs-{epoch + 1}.jpg')
        torch.save(model_eager.state_dict(), f'{chkpts}/model_orig_snapshot.pt')
        if ema_model:
            torch.save(ema_model.ema_model.state_dict(), f'{chkpts}/EMA_model_snapshot.pt')

    # Save models and the training cgeckpoints after the training
    torch.save(model_eager.state_dict(), f'{chkpts}/final_model_{epoch + 1}.pt')
    if ema_model:
        torch.save(ema_model.ema_model.state_dict(), f'{chkpts}/EMA_final_model_{epoch + 1}.pt')

    print(f'Saved the final model at\n\t{chkpts}/final_model_e{epoch + 1}_s{step + 1}.pt')
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'm_opt': m_opt.state_dict(),
        'm_scaler': m_scaler.state_dict() if m_scaler else None,
        'd_opt': d_opt.state_dict(),
        'd_scaler': d_scaler.state_dict() if d_opt else None,
        'm_sch': m_sch.state_dict() if m_sch else None,
        'd_sch': m_sch.state_dict() if d_sch else None,
    }
    torch.save(checkpoint, f'{chkpts}/final_chkpt_e{epoch + 1}_s{step + 1}.pt')

    # final reconstruction
    with torch.no_grad():
        x_recon, *_ = model(x_original)
    x_recon = unscale_tensor(x_recon).detach()
    save_grid_imgs(x_recon, max(x_recon.shape[0] // 4, 2), \
                   f'{img_folder}/FINAL_recon_imgs-{step + 1}-{epoch + 1}.jpg')

    return 0


# ===================================================================    
if __name__ == '__main__':
    import argparse

    arg_desc = '''\
        Training of AE image compressor for latent diffusion
        -------------------------------------------------------
                Please, provide name to the config file
        '''

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=arg_desc)

    parser.add_argument("-cfg", metavar="config file", help="Path to the config", required=True)
    args = parser.parse_args()
    cfg_file = args.cfg

    msg = """
    ==============================================================
        Training of VQ-AE image compressor for latent diffusion
     on ArtBench dataset (https://github.com/liaopeiyuan/artbench)
    ==============================================================   
    """
    print(msg)
    main(cfg_file)
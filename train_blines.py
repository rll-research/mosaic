import os
import json 
import yaml
import copy
import torch
import hydra
import random 
import argparse
import datetime
import pickle as pkl
import numpy as np
import torch.nn as nn
from os.path import join
import torch.nn.functional as F
# import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from mosaic.utils.lr_scheduler import build_scheduler
from einops import rearrange, reduce, repeat, parse_shape
from mosaic.models.discrete_logistic import DiscreteMixLogistic
from collections import defaultdict, OrderedDict
from hydra.utils import instantiate
from mosaic.datasets.multi_task_datasets import BatchMultiTaskSampler, DIYBatchSampler, collate_by_task # need for val. loader



torch.autograd.set_detect_anomaly(True)
import learn2learn as l2l
# for visualization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,3,1,1))
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,3,1,1))

class Trainer:
    def __init__(self, description="Default model trainer", allow_val_grad=False, hydra_cfg=None):
        assert hydra_cfg is not None, "Need to start with hydra-enabled yaml file!"
        self.config = hydra_cfg
        self.train_cfg = hydra_cfg.train_cfg
        # initialize device
        def_device = hydra_cfg.device if hydra_cfg.device != -1 else 0
        self._device = torch.device("cuda:{}".format(def_device))
        self._device_list = None
        self._allow_val_grad = allow_val_grad 
        # set of file saving
        assert os.path.exists(self.config.save_path), "Warning! Save path {} doesn't exist".format(self.config.save_path)
        if '/shared' in self.config.save_path:
            print("Warning! saving data to /shared folder \n")
        assert self.config.exp_name != -1, 'Specify an experiment name for log data!'

        append = "-Batch{}".format(int(self.config.bsize)) 
        self.config.exp_name += append

        save_dir = join(self.config.get('save_path', './'), str(self.config.exp_name))
        save_dir = os.path.expanduser(save_dir)
        self._save_fname = join(save_dir, 'model_save')
        self.save_dir = save_dir
        self._step = None

    def calculate_maml_loss(self, meta_model, model_inputs):
        device = self._device #orch.device("cuda:1") #self._device
        states, actions = model_inputs['states'], model_inputs['actions']
        images, context = model_inputs['images'], model_inputs['demo']
        aux = model_inputs['aux_pose']

        meta_model = meta_model.to(device)
        inner_iters = self.config.get('inner_iters', 1)
        l2error = torch.nn.MSELoss()
 
        bc_loss, aux_loss = [], []

        for task in range(states.shape[0]):
            learner = meta_model.module.clone()
            for _ in range(inner_iters):
                learner.adapt(\
                learner(None, context[task], learned_loss=True)['learned_loss'], allow_nograd=True, allow_unused=True)
            out = learner(states[task], images[task], ret_dist=False)
            l_aux = l2error(out['aux'], aux[task][None])[None]
            mu, sigma_inv, alpha = out['action_dist']
            action_distribution = DiscreteMixLogistic(mu[:-1], sigma_inv[:-1], alpha[:-1])
            l_bc = -torch.mean(action_distribution.log_prob(actions[task]))[None]
            #validation_loss = l_bc + l_aux
            #error += validation_loss / states.shape[0]
            bc_loss.append(l_bc)
            aux_loss.append(l_aux)

        #stats = {'l_bc': np.mean(bc_loss), 'aux_loss': np.mean(aux_loss)}
        #return error, stats
        #print(len(bc_loss), bc_loss[0].shape, aux_loss[0].shape)
        return torch.cat(bc_loss, dim=0), torch.cat(aux_loss, dim=0)

    def calculate_task_loss(self, model, task_inputs):
        """Assumes inputs are collated by task names already, organize things properly before feeding into the model s.t.
           for each batch input, the model does only one forward pass."""
        all_loss, all_stats = dict(), dict()
        device = self._device
        model_inputs = defaultdict(list)
        task_to_idx = dict()
        task_losses = OrderedDict()
        start = 0
        for idx, (task_name, inputs) in enumerate(task_inputs.items()):
            traj = inputs['traj']
            for key in ['states', 'actions', 'images', 'images_cp', 'aux_pose']:
                model_inputs[key].append( traj[key].to(device) )

            model_inputs['points'].append( traj['points'].to(device).long() )
            for key in ['demo', 'demo_cp']:
                model_inputs[key].append( inputs['demo_data'][key].to(device) )

            task_bsize  = traj['actions'].shape[0]
            task_to_idx[task_name] = [ start + i for i in range(task_bsize)]
            task_losses[task_name] = OrderedDict()
            start += task_bsize

        for key in model_inputs.keys():
            model_inputs[key] = torch.cat(model_inputs[key], dim=0)

        if self.config.gen_png and (not self.generated_png):
            self.generate_figure(model_inputs['images'], model_inputs['demo'], '/home/{}/one_shot_transformers/burner.png'.format(USER))
            self.generate_figure(model_inputs['images_cp'], model_inputs['demo_cp'], '/home/{}/one_shot_transformers/burner_aug.png'.format(USER))
            self.generated_png = True
        all_losses = dict()
        if self.config.use_maml:
            bc_loss, aux_loss = self.calculate_maml_loss(model, model_inputs) 
            all_losses["l_bc"] = bc_loss
            all_losses["l_aux"] = aux_loss
            all_losses["loss_sum"] = bc_loss + aux_loss
        else:
            out = model(
                images=model_inputs['images'],
                context=model_inputs['demo'],
                states=model_inputs['states'], 
                ret_dist=False)
            # forward & backward action pred
            actions =  model_inputs['actions']
            mu_bc, scale_bc, logit_bc = out['bc_distrib'] # mu_bc.shape: B, 7, 8, 4]) but actions.shape: B, 6, 8
            action_distribution       = DiscreteMixLogistic(mu_bc[:,:-1], scale_bc[:,:-1], logit_bc[:,:-1])
            act_prob                  = rearrange(- action_distribution.log_prob(actions), 'B n_mix act_dim -> B (n_mix act_dim)')


            all_losses["l_bc"]     = self.train_cfg.bc_loss_mult * torch.mean(act_prob, dim=-1)
            # compute inverse model density
            inv_distribution       = DiscreteMixLogistic(*out['inverse_distrib'])
            inv_prob               = rearrange(- inv_distribution.log_prob(actions), 'B n_mix act_dim -> B (n_mix act_dim)')
            all_losses["l_inv"]    = self.train_cfg.inv_loss_mult * torch.mean(inv_prob, dim=-1)

            if 'point_ll' in out:
                pnts = model_inputs['points']
                l_point = self.train_cfg.pnt_loss_mult * \
                    torch.mean(-out['point_ll'][range(pnts.shape[0]), pnts[:,-1,0], pnts[:,-1,1]], dim=-1)

                all_losses["point_loss"] = l_point

            # NOTE: the model should just output calculated rep-learning loss
            rep_loss               = torch.zeros_like(all_losses["l_bc"] )
            for k, v in out.items():
                if k in self.train_cfg.rep_loss_muls.keys():
                    v              = torch.mean(v, dim=-1) # just return size (B,) here
                    v              = v * self.train_cfg.rep_loss_muls.get(k, 0)
                    all_losses[k]  = v
                    rep_loss       = rep_loss + v
            all_losses["rep_loss"] = rep_loss
            all_losses["loss_sum"] = all_losses["l_bc"] + all_losses["l_inv"] + rep_loss

        # flatten here to avoid headache
        for (task_name, idxs) in task_to_idx.items():
            for (loss_name, loss_val) in all_losses.items():
                if len(loss_val.shape) > 0:
                    task_losses[task_name][loss_name] = torch.mean(loss_val[idxs])
        return task_losses

    def collect_stats(self, task_losses, raw_stats, prefix='train'):
        """ create/append to stats dict of a one-layer dict structure:
            {'task_name/loss_key': [..], 'loss_key/task_name':[...]}"""
        task_names = sorted(task_losses.keys())
        for task, stats in task_losses.items():
            # expects: {'button': {"loss_sum": 1, "l_bc": 1}} 
            for k, v in stats.items(): 
                for log_key in [ f"{prefix}/{task}/{k}", f"{prefix}/{k}/{task}" ]:
                    if log_key not in raw_stats.keys():
                        raw_stats[log_key] = []
                    raw_stats[log_key].append(self._loss_to_scalar(v))
            if "step" in raw_stats.keys():
                raw_stats["step"].append(int(self._step)) 
            else:
                raw_stats["step"] = [int(self._step)]
        tr_print = ""
        for i, task in enumerate(task_names): 
            tr_print += "[{0:<9}] l_tot: {1:.1f} l_bc: {2:.1f} l_aux: {3:.1f} l_aux: {4:.1f} ".format( \
                task, 
                raw_stats[f"{prefix}/{task}/loss_sum"][-1], 
                raw_stats[f"{prefix}/{task}/l_bc"][-1],
                raw_stats.get(f"{prefix}/{task}/point_loss",[0])[-1], 
                raw_stats.get(f"{prefix}/{task}/l_aux",[0])[-1]) 
            if i % 3 == 2: # use two lines to print 
                tr_print += "\n"

        return tr_print

    def train(self, model, weights_fn=None, save_fn=None, optim_weights=None):
        """New(0507): let's merge this function and train_fn for clarity """
        self._train_loader, self._val_loader = self._make_data_loaders(self.train_cfg)
        # wrap model in DataParallel if needed and transfer to correct device
        print('Training stage \n Found {} GPU devices \n'.format(self.device_count))
        model = model.to(self._device)
        if self.device_count > 1 and not isinstance(model, nn.DataParallel):
            print("Training stage \n Device list: {}".format(self.device_list))
            model = nn.DataParallel(model, device_ids=self.device_list)

        # initialize optimizer and lr scheduler
        optim_weights       = optim_weights if optim_weights is not None else model.parameters()
        optimizer, scheduler = self._build_optimizer_and_scheduler(optim_weights, self.train_cfg)

        # initialize constants:
        epochs              = self.train_cfg.get('epochs', 1)
        vlm_alpha           = self.train_cfg.get('vlm_alpha', 0.6)
        log_freq            = self.train_cfg.get('log_freq', 1000)
        val_freq            = self.train_cfg.get('val_freq', 1000)
        print_freq          = self.train_cfg.get('print_freq', 100)
        self._img_log_freq  = img_log_freq = self.train_cfg.get('img_log_freq', 10000)
        assert img_log_freq % log_freq == 0, "log_freq must divide img_log_freq!"
        save_freq           = self.config.get('save_freq', 10000)

        print("Loss multipliers: \n BC: {} inv: {} Point: {}".format(
            self.train_cfg.bc_loss_mult, self.train_cfg.inv_loss_mult, self.train_cfg.pnt_loss_mult))
        print({name: mul for name, mul in self.train_cfg.rep_loss_muls.items() if mul != 0})
        if self.train_cfg.bc_loss_mult == 0 and self.train_cfg.inv_loss_mult == 0:
            assert sum([v for k, v in self.train_cfg.rep_loss_muls.items()]) != 0, self.train_cfg.rep_loss_muls

        self.tasks          = self.config.tasks
        num_tasks           = len(self.tasks)
        sum_mul             = sum( [task.get('loss_mul', 1) for task in self.tasks] )
        task_loss_muls      = { task.name:
            float("{:3f}".format(task.get('loss_mul', 1) / sum_mul)) for task in self.tasks }
        print(" Weighting each task loss separately:", task_loss_muls)
        self.generated_png  = False
        self._step          = 0
        val_iter            = iter(self._val_loader) 
        # log stats to both 'task_name/loss_name' AND 'loss_name/task_name'
        raw_stats           = dict()
        print(f"Training for {epochs} epochs train dataloader has length {len(self._train_loader)}, \
                which sums to {epochs * len(self._train_loader)} total train steps, \
                validation loader has length {len(self._val_loader)}")
        for e in range(epochs):
            frac = e / epochs
            if self.config.use_byol:
                mod = model.module if isinstance(model, nn.DataParallel) else model
                mod.momentum_update(frac)

            for inputs in self._train_loader:

                if self._step % save_freq == 0: # save model AND stats
                    self.save_checkpoint(model, optimizer, weights_fn, save_fn)
                    if save_fn is not None:
                        save_fn(self._save_fname, self._step)
                    else:
                        save_module = model
                        if weights_fn is not None:
                            save_module = weights_fn()
                        elif isinstance(model, nn.DataParallel):
                            save_module = model.module
                        torch.save(save_module.state_dict(), self._save_fname + '-{}.pt'.format(self._step))
                    if self.config.get('save_optim', False):
                        torch.save(optimizer.state_dict(), self._save_fname + '-optim-{}.pt'.format(self._step))

                    stats_save_name = join(self.save_dir, 'stats', '{}.json'.format('train_val_stats'))
                    json.dump({k: str(v) for k, v in raw_stats.items()}, open(stats_save_name, 'w'))
                    
                optimizer.zero_grad()
                ## calculate loss here:
                task_losses = self.calculate_task_loss(model, inputs)
                task_names = sorted(task_losses.keys())
                weighted_task_loss = sum([l["loss_sum"] * task_loss_muls.get(name) for name, l in task_losses.items()])
                weighted_task_loss.backward()
                optimizer.step()
                ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
                # calculate train iter stats
                if self._step % log_freq == 0:
                    train_print = self.collect_stats(task_losses, raw_stats, prefix='train')   
                    
                    if self._step % print_freq == 0:
                        print('Training epoch {1}/{2}, step {0}: \t '.format(self._step, e, epochs))
                        print(train_print)
                        

                if self._step % val_freq == 0:
                    # exhaust all data in val loader and take avg loss
                    all_val_losses = {task: defaultdict(list) for task in task_names}
                    val_iter = iter(self._val_loader)
                    for val_inputs in val_iter:
                        if self.config.use_maml: # allow grad!
                            eval_model = nn.DataParallel(
                                model.eval().module, device_ids=self.device_list[1:])
                            val_task_losses = self.calculate_task_loss(eval_model, val_inputs)
                            
                        else:
                            with torch.no_grad():
                                model = model.eval()
                                val_task_losses = self.calculate_task_loss(model, val_inputs)
       
                        for task, losses in val_task_losses.items():
                            for k, v in losses.items():
                                all_val_losses[task][k].append(v)
                    # take average across all batches in the val loader 
                    avg_losses = dict()
                    for task, losses in all_val_losses.items():
                        avg_losses[task] = {
                            k: torch.mean(torch.stack(v)) for k, v in losses.items()}
                    
                    val_print = self.collect_stats(avg_losses, raw_stats, prefix='val')
                    if self._step % print_freq == 0:
                        print('Validation step {}:'.format(self._step))
                        print(val_print)
                    model = model.train()
                 
                # update target params
                # mod = model.module if isinstance(model, nn.DataParallel) else model
                # if self._step % self.train_cfg.target_update == 0:
                #     mod.soft_param_update()

                
                self._step += 1
        ## when all epochs are done, save model one last time
        self.save_checkpoint(model, optimizer, weights_fn, save_fn)

    def save_checkpoint(self, model, optimizer, weights_fn=None, save_fn=None):
        if save_fn is not None:
            save_fn(self._save_fname, self._step)
        else:
            save_module = model
            if weights_fn is not None:
                save_module = weights_fn()
            elif isinstance(model, nn.DataParallel):
                save_module = model.module
            torch.save(save_module.state_dict(), self._save_fname + '-{}.pt'.format(self._step))
        if self.config.get('save_optim', False):
            torch.save(optimizer.state_dict(), self._save_fname + '-optim-{}.pt'.format(self._step))
        print(f'Model checkpoint saved at step {self._step}')
        return 

    @property
    def device_count(self):
        if self._device_list is None:
            return torch.cuda.device_count()
        return len(self._device_list)

    @property
    def device_list(self):
        if self._device_list is None:
            return [i for i in range(torch.cuda.device_count())]
        return copy.deepcopy(self._device_list)

    @property
    def device(self):
        return copy.deepcopy(self._device)

    def _make_data_loaders(self, cfg):
        """
        use yaml cfg to return train and val dataloaders, NOTE: both train and val uses collate_by_task now
        """
        assert '_target_' in cfg.dataset.keys(), "Let's use hydra-config from now on. "
        print("Initializing {} with hydra config. \n".format(cfg.dataset._target_))
        #if cfg.dataset.get('agent_dir', None):
        #print("Agent file dirs: ", cfg.dataset.root_dir)
        cfg.dataset.mode = 'train'
        dataset = instantiate(cfg.dataset)
        samplerClass = DIYBatchSampler if cfg.sampler.use_diy else BatchMultiTaskSampler
        train_sampler = samplerClass(
                task_to_idx=dataset.task_to_idx,
                subtask_to_idx=dataset.subtask_to_idx,
                tasks_spec=cfg.dataset.tasks_spec,
                sampler_spec=cfg.sampler)
        #print("Dataloader has batch size {} \n".format(cfg.batch_size))
        train_loader = DataLoader(
            dataset,
            batch_sampler=train_sampler,
            num_workers=min(11, self.config.get('loader_workers', cpu_count())),
            worker_init_fn=lambda w: np.random.seed(np.random.randint(2 ** 29) + w),
            collate_fn=collate_by_task
            )
        #print("Train loader has {} iterations".format(len(train_loader)))

        cfg.dataset.mode = 'val'
        val_dataset = instantiate(cfg.dataset)
        cfg.sampler.batch_size = cfg.val_size # allow validation batch to have a different size
        val_sampler = samplerClass(
                task_to_idx=val_dataset.task_to_idx,
                subtask_to_idx=val_dataset.subtask_to_idx,
                tasks_spec=cfg.dataset.tasks_spec,
                sampler_spec=cfg.sampler,)

        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=min(11, self.config.get('loader_workers', cpu_count())),
            worker_init_fn=lambda w: np.random.seed(np.random.randint(2 ** 29) + w),
            collate_fn=collate_by_task
            )
        #print("Validation loader has {} total samples".format(len(val_loader)))
        return train_loader, val_loader

    def _build_optimizer_and_scheduler(self, optim_weights, cfg):
        assert self.device_list is not None, str(self.device_list)
        optimizer = torch.optim.Adam(
            optim_weights, cfg.lr, weight_decay=cfg.get('weight_decay', 0))
        return optimizer, build_scheduler(optimizer, cfg.get('lr_schedule', {}))

    def _step_optim(self, loss, step, optimizer):
        loss.backward()
        optimizer.step()

    def _zero_grad(self, optimizer):
        optimizer.zero_grad()

    def _loss_to_scalar(self, loss):
        """New(0511): cut down precision here just for logging purpose"""
        x = loss.item()
        return float("{:.3f}".format(x))

    @property
    def step(self):
        if self._step is None:
            raise Exception("Optimization has not begun!")
        return self._step

    @property
    def is_img_log_step(self):
        return self._step % self._img_log_freq == 0

    def generate_figure(self, images, context, fname='burner.png'):
        _B, T_im, _, _H, _W = images.shape
        T_con = context.shape[1]
        print("Images value range: ", images.min(), images.max(), context.max())
        print("Generating figures from images shape {}, context shape {} \n".format(images.shape, context.shape))
        npairs = 7
        skip   = 8
        ncols  = 4
        fig, axs = plt.subplots(nrows=npairs * 2, ncols=ncols, figsize=(ncols*3.5, npairs*2*2.8), subplot_kw={'xticks': [], 'yticks': []})
        fig.subplots_adjust(left=0.03, right=0.97, hspace=0.3, wspace=0.05)
        for img_index in range(npairs):
            show_img = images[img_index*skip].cpu().numpy() * STD + MEAN
            show_con = context[img_index*skip].cpu().numpy() * STD + MEAN
            for count in range(ncols):
                axs[img_index*2, count].imshow(show_img[count].transpose(1,2,0))
                if count < T_con:
                    axs[img_index*2+1, count].imshow(show_con[count].transpose(1,2,0))

        plt.tight_layout()
        print("Saving figure to: ", fname)
        plt.savefig(fname)

class Workspace(object):
    """
    Initializes the action model;
    defines how to caculate losses based on the model's output;
    make them the output of train function function;
    provide to the Trainer class above
    """
    def __init__(self, cfg):
        self.trainer = Trainer(allow_val_grad=False, hydra_cfg=cfg)
        print("Finished initializing trainer")
        config = self.trainer.config
        train_cfg = config.train_cfg
        resume = config.get('resume', False)
        self.action_model = hydra.utils.instantiate(config.policy)
        if config.use_maml:
            print("Switching to l2l.algorithms.MAML")
            self.action_model = l2l.algorithms.MAML(
                self.action_model,
                lr=config['policy']['maml_lr'],
                first_order=config['policy']['first_order'],
                allow_unused=True)


        print("Action model initialized to: {}".format(config.policy._target_))
        if resume:
            rpath = os.path.join('/home/{}/one_shot_transformers/baseline_data'.format(USER), config.resume_path)
            if not os.path.exists(rpath): # on undergrad servers
                rpath = os.path.join('/home/{}/osil'.format(USER), config.resume_path)
            if not os.path.exists(rpath):
                rpath = os.path.join('/shared/{}/bline_osil'.format(USER), config.resume_path)
            if not os.path.exists(rpath):
                rpath = join('/home/{}/2021/NeurIPS/one_shot_transformers/log_data'.format(USER), config.resume_path)
            assert os.path.exists(rpath), "Can't seem to find {} anywhere".format(config.resume_path)
            print('load model from ...%s' % rpath)

            self.action_model.load_state_dict(torch.load(rpath, map_location=torch.device('cpu')))
        self.config = config
        self.train_cfg = config.train_cfg

        ## move log path to here!
        print('\n Done initializing Workspace, saving config.yaml to directory: {}'.format(self.trainer.save_dir))

        os.makedirs(self.trainer.save_dir, exist_ok=('burn' in self.trainer.save_dir))
        os.makedirs(join(self.trainer.save_dir, 'stats'), exist_ok=True)
         
        save_config = copy.deepcopy(self.trainer.config)
        OmegaConf.save(config=save_config, f=join(self.trainer.save_dir, 'config.yaml'))


    def run(self):
        mod = self.action_model.module if isinstance(self.action_model, nn.DataParallel) else self.action_model
        if self.config.freeze_img_encoder:
            print("Freezing image encoder:")
            mod.freeze_img_encoder()
        if self.config.freeze_attention:
            print("Freezing transformer layers:")
            mod.freeze_attn_layers(self.config.num_freeze_layers)
        if self.config.restart_action_layers:
            print("Switching to new action head")
            mod.restart_action_layers()
        if self.config.train_encoder_only:
            print("Freezing the attention and action heads to train only the image encoder")
            mod.pretrain_img_encoder()

        self.trainer.train(self.action_model)
        print("Done training")


@hydra.main(
    config_path="experiments", 
    config_name="baselines.yaml")
def main(cfg):
    #print(cfg)
    from train_blines import Workspace as W
     
    if cfg.use_lstm:
        print("Switching to LSTM model for MT-dataset")
        cfg.policy = copy.deepcopy(cfg.lstm_policy)
    if cfg.use_maml:
        print("Switching to MAML for MT-dataset")
        cfg.policy = copy.deepcopy(cfg.maml_policy)
    
    if cfg.single_task:
        cfg.tasks = [cfg.single_task]
    if cfg.use_all_tasks:
        print("New(0508): loading setting all 7 tasks to the dataset!  obs_T: {} demo_T: {}".format(\
            cfg.dataset_cfg.obs_T, cfg.dataset_cfg.demo_T))
        cfg.tasks = [cfg.nut_assembly, cfg.door, cfg.drawer, cfg.button, cfg.new_pick_place, cfg.stack_block, cfg.basketball]
     
    if cfg.set_same_n > -1:
        print('New(0514): setting n_per_task of all tasks to ', cfg.set_same_n)
        for tsk in cfg.tasks:
            tsk.n_per_task = cfg.set_same_n
    if cfg.limit_num_traj > -1:
        print('New(0521): only uses {} trajectory for each sub-task'.format(cfg.limit_num_traj))
        for tsk in cfg.tasks:
            tsk.traj_per_subtask = cfg.limit_num_traj
    if cfg.limit_num_demo > -1:
        print('New(0521): only uses {} demon. trajectory for each sub-task'.format(cfg.limit_num_demo))
        for tsk in cfg.tasks:
            tsk.demo_per_subtask = cfg.limit_num_demo 
    if cfg.weight_loss_by_subtask:
        print("New(0514): setting the loss multipliers for each task to equal its num of subtasks")
        for tsk in cfg.tasks:
            tsk.loss_mul = tsk.n_tasks

    workspace = W(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

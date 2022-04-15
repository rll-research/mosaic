import os
import json 
import copy
import torch
import hydra  
import numpy as np
import torch.nn as nn
from os.path import join
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
from omegaconf import OmegaConf 
from mosaic.utils.lr_scheduler import build_scheduler
from train_utils import generate_figure
from einops import rearrange
from mosaic.models.discrete_logistic import DiscreteMixLogistic
from collections import defaultdict, OrderedDict
from hydra.utils import instantiate
from mosaic.datasets.multi_task_datasets import DIYBatchSampler, collate_by_task 
torch.autograd.set_detect_anomaly(True)

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
        assert self.config.exp_name != -1, 'Specify an experiment name for log data!'
        
        append = "-Batch{}-{}gpu-Attn{}ly{}-Act{}ly{}mix{}".format(
            int(self.config.bsize), int(torch.cuda.device_count()),
            int(self.config.policy.attn_cfg.n_attn_layers), int(self.config.policy.attn_cfg.attn_ff),
            int(self.config.policy.action_cfg.n_layers), int(self.config.policy.action_cfg.out_dim),
            int(self.config.policy.action_cfg.n_mixtures))
        #print(self.config.policy)
        if self.config.policy.concat_demo_head: 
            append += "-headCat"
        elif self.config.policy.concat_demo_act:
            append += "-actCat"
        else:
            append += "-noCat"
        append += "-simclr{}x{}".format(int(self.config.policy.simclr_config.compressor_dim), int(self.config.policy.simclr_config.hidden_dim))
        self.config.exp_name += append 

        save_dir = join(self.config.get('save_path', './'), str(self.config.exp_name))
        save_dir = os.path.expanduser(save_dir)
        self._save_fname = join(save_dir, 'model_save')
        self.save_dir = save_dir
        self._step = None

    def calculate_task_loss(self, model, task_inputs):
        """Assumes inputs are collated by task names already, organize things properly before feeding into the model s.t.
           for each batch input, the model does only one forward pass."""
        device = self._device
        model_inputs = defaultdict(list)
        task_to_idx = dict()
        task_losses = OrderedDict()
        start = 0
        for idx, (task_name, inputs) in enumerate(task_inputs.items()):
            traj = inputs['traj']
            for key in ['states', 'actions', 'images', 'images_cp']:
                model_inputs[key].append( traj[key].to(device) )
            for key in ['demo', 'demo_cp']:
                model_inputs[key].append( inputs['demo_data'][key].to(device) )
            
            task_bsize  = traj['actions'].shape[0]
            task_to_idx[task_name] = [ start + i for i in range(task_bsize)]
            task_losses[task_name] = OrderedDict()
            start += task_bsize
        
        for key in ['states', 'actions', 'images', 'images_cp'] + ['demo', 'demo_cp']:
            model_inputs[key] = torch.cat(model_inputs[key], dim=0)

        if self.config.gen_png and (not self.generated_png):
            image_path = join(self.config.save_path, 'input_batch.png')
            print('Generating input batch image: {}'.format(image_path))
            generate_figure(model_inputs['images'], model_inputs['demo'], image_path)
            self.generated_png = True
 
        out = model(
            images=model_inputs['images'], images_cp=model_inputs['images_cp'], 
            context=model_inputs['demo'],  context_cp=model_inputs['demo_cp'],
            states=model_inputs['states'], ret_dist=False,
            actions=model_inputs['actions']
            ) 
        all_losses = dict()

        # forward & backward action pred
        actions                   = model_inputs['actions']
        mu_bc, scale_bc, logit_bc = out['bc_distrib'] # mu_bc.shape: B, 7, 8, 4]) but actions.shape: B, 6, 8
        action_distribution       = DiscreteMixLogistic(mu_bc[:,:-1], scale_bc[:,:-1], logit_bc[:,:-1])
        act_prob                  = rearrange(- action_distribution.log_prob(actions), 'B n_mix act_dim -> B (n_mix act_dim)')
            
        all_losses["l_bc"]     = self.train_cfg.bc_loss_mult * torch.mean(act_prob, dim=-1)
        
        # compute inverse model density
        inv_distribution       = DiscreteMixLogistic(*out['inverse_distrib'])
        inv_prob               = rearrange(- inv_distribution.log_prob(actions), 'B n_mix act_dim -> B (n_mix act_dim)')
        all_losses["l_inv"]    = self.train_cfg.inv_loss_mult * torch.mean(inv_prob, dim=-1)
        
        # NOTE: action loss is computed here, but the model should output contrastive losses 
        rep_loss               = torch.zeros_like(all_losses["l_bc"])
        for k, v in out.items():
            if k in self.train_cfg.rep_loss_muls.keys():
                v              = torch.mean(v, dim=-1) # just return size (B,) here 
                v              = v * self.train_cfg.rep_loss_muls.get(k, 0)
                all_losses[k]  = v
                rep_loss       += v
        all_losses["rep_loss"] = rep_loss 
        all_losses["loss_sum"] = all_losses["l_bc"] + all_losses["l_inv"] + rep_loss
        for (task_name, idxs) in task_to_idx.items():
            for (loss_name, loss_val) in all_losses.items():
                if len(loss_val.shape) > 0:
                    task_losses[task_name][loss_name] = torch.mean(loss_val[idxs]) 
            
        return task_losses 

    def collect_stats(self, task_losses, raw_stats, running_means=None):
        for name, stats in task_losses.items():
            # expects: {'task_name': {"loss_sum": 1, "l_bc": 1}}
            assert name in raw_stats.keys(), 'Got unexpected task name ' + str(name)
            for k, v in stats.items():
                if k not in raw_stats[name].keys():
                    raw_stats[name][k] = [] 
                raw_stats[name][k].append(self._loss_to_scalar(v))
            raw_stats[name]["step"].append(int(self._step))
        tr_print = ""
        for i, (task, v) in enumerate(raw_stats.items()):
            tr_print += "[{0:<9}] l_tot: {1:.2f} l_bc: {2:.2f} l_rep: {3:.2f}  ".format( \
                task, v["loss_sum"][-1], v["l_bc"][-1], v["rep_loss"][-1])
            if running_means:
                tr_print += " vl_mean: {:.2f}  ".format(running_means.get(task, 0))
            if i % 3 == 2:
                tr_print += "\n"
            
        return tr_print 

    def train(self, model, weights_fn=None, save_fn=None, optim_weights=None): 
        self._train_loader, self._val_loader = self._make_data_loaders(self.train_cfg)
        # wrap model in DataParallel if needed and transfer to correct device
        print('Begin training: \n Found {} GPU devices \n'.format(self.device_count))
        model = model.to(self._device)
        if self.device_count > 1 and not isinstance(model, nn.DataParallel):
            print("Begin training: \n Device list: {}".format(self.device_list))
            model = nn.DataParallel(model, device_ids=self.device_list)

        # initialize optimizer and lr scheduler
        optim_weights       = optim_weights if optim_weights is not None else model.parameters()
        optimizer, scheduler = self._build_optimizer_and_scheduler(optim_weights, self.train_cfg)

        # initialize constants:
        epochs              = self.train_cfg.get('epochs', 1)
        vlm_alpha           = self.train_cfg.get('vlm_alpha', 0.6)
        log_freq            = self.train_cfg.get('log_freq', 1000)
        print_freq          = self.train_cfg.get('print_freq', 100)
        save_freq           = self.config.get('save_freq', 10000)

        print("Loss multipliers: \n BC: {} inv: {} Point: {}".format(
            self.train_cfg.bc_loss_mult, self.train_cfg.inv_loss_mult, self.train_cfg.pnt_loss_mult))
        print({name: mul for name, mul in self.train_cfg.rep_loss_muls.items() if mul != 0})
        if self.train_cfg.bc_loss_mult == 0 and self.train_cfg.inv_loss_mult == 0:
            assert sum([v for k, v in self.train_cfg.rep_loss_muls.items()]) != 0, self.train_cfg.rep_loss_muls
        
        self.tasks          = self.config.tasks

        sum_mul             = sum( [task.get('loss_mul', 1) for task in self.tasks] )
        task_loss_muls      = { task.name: 
            float("{:3f}".format(task.get('loss_mul', 1) / sum_mul)) for task in self.tasks }
        print("Weighting each task loss:", task_loss_muls)
        self.generated_png  = False
        self._step          = 0
        val_iter            = iter(self._val_loader)
        raw_val_stats       = OrderedDict({ task.name: dict({"step": []}) for task in self.tasks })
        raw_train_stats     = OrderedDict({ task.name: dict({"step": []}) for task in self.tasks })
        vl_running_means    = OrderedDict({ task.name: 0 for task in self.tasks } )
        for e in range(epochs):
            frac = e / epochs  
            mod = model.module if isinstance(model, nn.DataParallel) else model 
            mod.momentum_update(frac)
            
            for inputs in self. _train_loader:
                optimizer.zero_grad() 
                task_losses = self.calculate_task_loss(model, inputs)
                weighted_task_loss = sum([l["loss_sum"] * task_loss_muls.get(name) for name, l in task_losses.items()]) 
                weighted_task_loss.backward()
                optimizer.step()
                ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## ## 
                # calculate stats
                mod_step = self._step % log_freq
                if mod_step == 0:
                    train_print = self.collect_stats(task_losses, raw_train_stats)
                    try:
                        val_inputs = next(val_iter)
                    except StopIteration:
                        val_iter = iter(self._val_loader)
                        val_inputs = next(val_iter)
                    with torch.no_grad():
                        model = model.eval()
                        val_task_losses = self.calculate_task_loss(model, val_inputs)
                        model = model.train()
                    val_print = self.collect_stats(val_task_losses, raw_val_stats, vl_running_means)
                    # update running mean stat
                    for name, stats in raw_val_stats.items():
                        vl_running_means[name] = stats["l_bc"][-1] * vlm_alpha + vl_running_means[name] * (1 - vlm_alpha)
 
                    print('Training epoch {1}/{2} \n, step {0}: \t '.format(self._step, e, epochs))
                    print(train_print)
                    print('Validation losses: \n', val_print)
                    
                elif self._step % print_freq == 0:
                    running = ""
                    for k, l in task_losses.items():
                        running += "[{}: {:.2f}] ".format(k, l["l_bc"].item())
                    print('step:', self._step, running, end='\r')
                
                self._step += 1
                # update target params
                mod = model.module if isinstance(model, nn.DataParallel) else model 
                if self._step % self.train_cfg.target_update == 0:
                    mod.soft_param_update()
                    
                if self._step % save_freq == 0:
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
                    
                    for name, log_stats in zip(\
                        ['train_stats', 'val_stats', 'vl_means'], [raw_train_stats, raw_val_stats, vl_running_means]):
                        stats_save_name = join(self.save_dir, 'stats', '{}.json'.format(name))
                        json.dump({k: str(v) for k, v in log_stats.items()}, open(stats_save_name, 'w'))
          
        ## when all epochs are done, save model one last time
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
        """ Use .yaml cfg to create both train and val dataloaders """
        print("Initializing {} with hydra config. \n".format(cfg.dataset._target_))
        cfg.dataset.mode = 'train'
        dataset = instantiate(cfg.dataset)
        train_sampler = DIYBatchSampler(
                task_to_idx=dataset.task_to_idx,
                subtask_to_idx=dataset.subtask_to_idx,
                tasks_spec=cfg.dataset.tasks_spec,
                sampler_spec=cfg.sampler)

        train_loader = DataLoader(
            dataset, 
            batch_sampler=train_sampler,
            num_workers=min(11, self.config.get('loader_workers', cpu_count())),
            worker_init_fn=lambda w: np.random.seed(np.random.randint(2 ** 29) + w),
            collate_fn=collate_by_task
            )

        cfg.dataset.mode = 'val'
        val_dataset = instantiate(cfg.dataset)
        cfg.sampler.batch_size = cfg.val_size # allow validation batch to have a different size
        val_sampler = DIYBatchSampler(
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

        return train_loader, val_loader

    def _build_optimizer_and_scheduler(self, optim_weights, cfg):
        assert self.device_list is not None, str(self.device_list)
        optimizer = torch.optim.Adam(
            optim_weights, cfg.lr, weight_decay=cfg.get('weight_decay', 0))
        return optimizer, build_scheduler(optimizer, cfg.get('lr_schedule', {}))

    def _loss_to_scalar(self, loss):
        """For more readable logging"""
        x = loss.item()
        return float("{:.3f}".format(x))

class Workspace(object):
    """ Initializes the policy model and prepare for Trainer.train() """
    def __init__(self, cfg):
        resume = cfg.get('resume', False)
        if resume: 
            rpath = join(cfg.save_path, cfg.resume_path) 
            assert os.path.exists(rpath), "Can't seem to find {} anywhere".format(cfg.resume_path)
            print('load model checkpoint AND model config from: %s' % rpath) 
            saved_yaml = OmegaConf.load(rpath.replace(cfg.resume_path.split('/')[-1], 'config.yaml'))
            self.action_model = hydra.utils.instantiate(saved_yaml.policy) 
            cfg.policy = copy.deepcopy(saved_yaml.policy)
            cfg.actions = copy.deepcopy(saved_yaml.policy.action_cfg)
            cfg.attn = copy.deepcopy(saved_yaml.policy.attn_cfg)
            
        self.trainer = Trainer(allow_val_grad=False, hydra_cfg=cfg)
        print("Finished initializing Trainer")
        config = self.trainer.config 
        
        self.action_model = hydra.utils.instantiate(config.policy)
        print("Action model initialized to: {}".format(config.policy._target_))
        if resume: 
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
        if self.config.freeze_attn_layers > -1 :
            print("Freezing transformer layers:")
            mod.freeze_attn_layers(int(self.config.freeze_attn_layers))
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
    config_name="multi_task_configs.yaml")
def main(cfg):
    from train import Workspace as W
    if cfg.use_all_tasks:
        print("Loading all 7 tasks to the dataset!  obs_T: {} demo_T: {}".format(\
            cfg.dataset_cfg.obs_T, cfg.dataset_cfg.demo_T))
        cfg.tasks = [cfg.nut_assembly, cfg.door, cfg.drawer, cfg.button, cfg.pick_place, cfg.stack_block, cfg.basketball]
    if cfg.set_same_n > -1:
        print('To construct a batch, setting n_per_task of all tasks to ', cfg.set_same_n)
        for tsk in cfg.tasks:
            tsk.n_per_task = cfg.set_same_n
    if cfg.limit_num_traj > -1:
        print('Only uses {} trajectory for each sub-task'.format(cfg.limit_num_traj))
        for tsk in cfg.tasks:
            tsk.traj_per_subtask = cfg.limit_num_traj
    if cfg.limit_num_demo > -1:
        print('Only uses {} demonstration trajectory for each sub-task'.format(cfg.limit_num_demo))
        for tsk in cfg.tasks:
            tsk.demo_per_subtask = cfg.limit_num_demo 

    
    workspace = W(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

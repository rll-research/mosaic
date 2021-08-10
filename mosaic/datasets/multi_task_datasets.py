import random
import os
import json
import torch
from os.path import join, expanduser
from moaic.datasets import load_traj, split_files

from torch.utils.data import Dataset, Sampler, DataLoader, SubsetRandomSampler, RandomSampler
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms
from torchvision.transforms import RandomAffine, ToTensor, Normalize, \
    RandomGrayscale, ColorJitter, RandomApply, RandomHorizontalFlip, GaussianBlur, RandomResizedCrop
from torchvision.transforms import functional as TvF
from torchvision.transforms.functional import resized_crop

import pickle as pkl
from collections import defaultdict, OrderedDict 
import glob 
import matplotlib.pyplot as plt
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape((1,3,1,1))
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape((1,3,1,1))
JITTER_FACTORS = {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1} 

def collate_by_task(batch):
    """Use this for validation: groups data by task names, so we
    can get per-task losses """
    per_task_data = defaultdict(list)
    for b in batch:
        per_task_data[b['task_name']].append(
            {k:v for k, v in b.items() if k != 'task_name' and k != 'task_id'}
        )

    for name, data in per_task_data.items():
        per_task_data[name] = default_collate(data)
    return per_task_data

class MultiTaskPairedDataset(Dataset):
    def __init__(
        self,
        tasks_spec,
        root_dir='/home/mandi/robosuite/multi_task',
        mode='train',
        split=[0.9, 0.1],
        demo_T=4,
        obs_T=7,
        aug_twice=True,
        width=180,
        height=100,
        data_augs=None,
        non_sequential=False,
        state_spec=('ee_aa', 'gripper_qpos'),
        action_spec=('action',),
        allow_val_skip=False,
        allow_train_skip=False,
        use_strong_augs=False,
        aux_pose=False,
        **params):
        """
        Args:
        -  root_dir:
            path to robosuite multi-task's data folder e.g. /home/mandi/robosuite/multi_task
        -  tasks_spec:
            a **List** specifying data location and other info for each task
            e.g. task_name = 'place'
                tasks_spec[0] = {
                    'name':             'place'
                    'date':             '0418'
                    'crop':             [30, 70, 85, 85], # this is pre-data-aug cropping
                    'traj_per_subtask': 100,
                    'n_tasks':          16,
                    'skip_id':          [-1] # a list of task ids to exclude!
                }
        (below are info shared across tasks:)
        - height， width
            crop params can be different but keep final image sizes the same
        - demo_T, obs_T:
            fixed demontration length and observation length
        - data_augs:
            specify how to crop/translate/jitter the data _after_ each image is cropped into the same sizes
            e.g. {
                'rand_trans': 0.1,      # proportionally shift the image by 0.1 times its height/width
                'jitter': 0.5,    # probability _factor_ to jitter each of the four jitter factors
                'grayscale': 0.2,       # got recommended 0.2 or 0.1
                }
        - state_spec:
            which state vectors to extract
                e.g. ('ee_aa', 'ee_vel', 'joint_pos', 'joint_vel', 'gripper_qpos', 'object_detected')
        -  action_spec
                action keys to get
        -  allow_train_skip, allow_val_skip:
                whether we entirely skip loading some of the subtasks to the dataset
        -   non_sequential：
                whether to take strides when we sample， note if we do this then inverse dynamics loss is invalid
        """
        self.task_crops     = OrderedDict()
        self.all_file_pairs = OrderedDict() # each idx i maps to a unique tuple of (task_name, sub_task_id, agent.pkl, demo.pkl)
        count               = 0
        self.task_to_idx    = defaultdict(list)
        self.subtask_to_idx = OrderedDict()
        for spec in tasks_spec:
            name, date      = spec.get('name', None), spec.get('date', None)
            assert (name and date), 'need to specify which day the data was generated, for easier tracking'
            if mode == 'train':
                print("Loading task [{:<9}] saved on date {}".format(name, date))
            agent_dir       = join(root_dir, name, '{}_panda_{}'.format(date, name))
            demo_dir        = join(root_dir, name, '{}_sawyer_{}'.format(date, name))
            self.subtask_to_idx[name] = defaultdict(list)
            for _id in range(spec.get('n_tasks')):
                if _id in spec.get('skip_ids', []):
                    if (allow_train_skip and mode == 'train') or (allow_val_skip and mode == 'val'):
                        print('Warning! Excluding subtask id {} from loaded **{}** dataset for task {}'.format(_id, mode, name))
                        continue
                task_id     = 'task_{:02d}'.format(_id)
                task_dir    = expanduser(join(agent_dir,  task_id, '*.pkl'))
                agent_files = sorted(glob.glob(task_dir))
                assert len(agent_files) != 0, "Can't find dataset for task {}, subtask {}".format(name, _id)
                subtask_size = spec.get('traj_per_subtask', 100)
                assert len(agent_files) >= subtask_size, "Doesn't have enough data "+str(len(agent_files))
                agent_files = agent_files[:subtask_size]

                ## prev. version does split randomly, here we strictly split each subtask in the same split ratio:
                idxs        = split_files(len(agent_files), split, mode)
                agent_files = [agent_files[i] for i in idxs]

                task_dir    = expanduser(join(demo_dir, task_id, '*.pkl'))

                demo_files  = sorted(glob.glob(task_dir))
                subtask_size = spec.get('demo_per_subtask', 100)
                assert len(demo_files) >= subtask_size, "Doesn't have enough data "+str(len(demo_files))
                demo_files  = demo_files[:subtask_size]
                idxs        = split_files(len(demo_files), split, mode)
                demo_files  = [demo_files[i] for i in idxs]
                # assert len(agent_files) == len(demo_files), \
                #     'data for task {}, subtask #{} is not matched'.format(name, task_id)

                for demo in demo_files:
                    for agent in agent_files:
                        self.all_file_pairs[count] = (name, _id, demo, agent)
                        self.task_to_idx[name].append(count)
                        self.subtask_to_idx[name][task_id].append(count)
                        count += 1
            #print('Done loading Task {}, agent/demo trajctores pairs reach a count of: {}'.format(name, count))
            self.task_crops[name] = spec.get('crop', [0,0,0,0])
        self.pairs_count = count
        self.task_count = len(tasks_spec)

        self._demo_T, self._obs_T = demo_T, obs_T
        self.width, self.height = width, height
        self.aug_twice = aug_twice
        self.aux_pose  = aux_pose

        self._state_action_spec = (state_spec, action_spec)
        self.non_sequential = non_sequential
        if non_sequential:
            print("Warning! The agent observations are not sampled in neighboring timesteps, make sure inverse dynamics loss is NOT used in training \n ")

        assert data_augs, 'Must give some basic data-aug parameters'
        if mode == 'train':
            print('Data aug parameters:', data_augs)
        #self.randAffine = RandomAffine(degrees=0, translate=(data_augs.get('rand_trans', 0.1), data_augs.get('rand_trans', 0.1)))
        self.toTensor = ToTensor()
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        jitters = {k: v * data_augs.get('weak_jitter', 0) for k,v in JITTER_FACTORS.items()}
        weak_jitter = ColorJitter(**jitters)

        weak_scale = data_augs.get('weak_crop_scale', (0.8, 1.0))
        weak_ratio = data_augs.get('weak_crop_ratio', (1.6, 1.8))
        randcrop   = RandomResizedCrop(size=(height, width), scale=weak_scale, ratio=weak_ratio)
        if data_augs.use_affine:
            randcrop = RandomAffine(degrees=0, translate=(data_augs.get('rand_trans', 0.1), data_augs.get('rand_trans', 0.1)))
        self.transforms = transforms.Compose([ # normalize at the end
            RandomApply([weak_jitter], p=0.1),
            RandomApply(
                [GaussianBlur(kernel_size=5, sigma=data_augs.get('blur', (0.1, 2.0))) ], p=0),
            randcrop,
            self.normalize])


        self.use_strong_augs = use_strong_augs
        print("Using strong augmentations?", use_strong_augs)
        jitters = {k: v * data_augs.get('strong_jitter', 0) for k,v in JITTER_FACTORS.items()}
        strong_jitter = ColorJitter(**jitters)
        self.grayscale = RandomGrayscale(data_augs.get("grayscale", 0))
        strong_scale = data_augs.get('strong_crop_scale', (0.2, 0.76))
        strong_ratio = data_augs.get('strong_crop_ratio', (1.2, 1.8))
        self.strong_augs = transforms.Compose([
            RandomApply([strong_jitter], p=0.05),
            self.grayscale,
            RandomHorizontalFlip(p=data_augs.get('flip', 0)),
            RandomApply(
                [GaussianBlur(kernel_size=5, sigma=data_augs.get('blur', (0.1, 2.0))) ], p=0.01),
            RandomResizedCrop(
                size=(height, width), scale=strong_scale, ratio=strong_ratio),
            self.normalize,
            ])

        def frame_aug(task_name, obs, second=False):
            """applies to every timestep's RGB obs['image']"""
            crop_params = self.task_crops.get(task_name, [0,0,0,0])
            top, left = crop_params[0], crop_params[2]
            img_height, img_width = obs.shape[0], obs.shape[1]
            box_h, box_w = img_height - top - crop_params[1], img_width - left - crop_params[3]

            obs = self.toTensor(obs)
            # only this resize+crop is task-specific
            obs = resized_crop(obs, top=top, left=left, height=box_h, width=box_w, size=(self.height, self.width))

            if self.use_strong_augs and second:
                augmented = self.strong_augs(obs)
            else:
                augmented = self.transforms(obs)
            assert augmented.shape == obs.shape

            return augmented
        self.frame_aug = frame_aug


    def __len__(self):
        """NOTE: we should count total possible demo-agent pairs, not just single-file counts
        total pairs should sum over all possible sub-task pairs"""
        return self.pairs_count

    def __getitem__(self, idx):
        """since the data is organized by task, use a mapping here to convert
        an index to a proper sub-task index """
        (task_name, sub_task_id, demo_file, agent_file) = self.all_file_pairs[idx]
        # print("getting idx", idx, task_name, sub_task_id)
        demo_traj, agent_traj = load_traj(demo_file), load_traj(agent_file)
        #assert (len(demo_traj) > 10 and len(agent_traj) > 10), \
        #    "Might have loaded in broken datafiles {}, length {}, {}, length {} ".format(demo_file, agent_file, len(demo_traj), len(agent_traj)=
        demo_data = self._make_demo(demo_traj, task_name)
        traj = self._make_traj(agent_traj, task_name)
        return {'demo_data': demo_data, 'traj': traj, 'task_name': task_name, 'task_id': sub_task_id}

    def _make_demo(self, traj, task_name):
        """
        Do a near-uniform sampling of the demonstration trajectory
        """
        clip = lambda x : int(max(0, min(x, len(traj) - 1)))
        per_bracket = max(len(traj) / self._demo_T, 1)
        frames = []
        cp_frames = []
        for i in range(self._demo_T):
            # fix to using uniform + 'sample_side' now
            if i == self._demo_T - 1:
                n = len(traj) - 1
            elif i == 0:
                n = 0
            else:
                n = clip(np.random.randint(int(i * per_bracket), int((i + 1) * per_bracket)))
            #frames.append(_make_frame(n))
            obs = traj.get(n)['obs']['image']

            processed = self.frame_aug(task_name, obs)
            frames.append(processed)
            if self.aug_twice:
                cp_frames.append(self.frame_aug(task_name, obs, True))

        ret_dict = dict()
        ret_dict['demo'] = torch.stack(frames)
        ret_dict['demo_cp'] = torch.stack(cp_frames)
        return ret_dict

    def _make_traj(self, traj, task_name):
        crop_params = self.task_crops.get(task_name, [0,0,0,0])
        def _adjust_points(points, frame_dims):
            h = np.clip(points[0] - crop_params[0], 0, frame_dims[0] - crop_params[1])
            w = np.clip(points[1] - crop_params[2], 0, frame_dims[1] - crop_params[3])
            h = float(h) / (frame_dims[0] - crop_params[0] - crop_params[1]) * self.height
            w = float(w) / (frame_dims[1] - crop_params[2] - crop_params[3]) * self.width
            return tuple([int(min(x, d - 1)) for x, d in zip([h, w], (self.height, self.width))])

        def _get_tensor(k, step_t):
            if k == 'action':
                return step_t['action']
            elif k == 'grip_action':
                return [step_t['action'][-1]]
            o = step_t['obs']
            if k == 'ee_aa' and 'ee_aa' not in o:
                ee, axis_angle = o['ee_pos'][:3], o['axis_angle']
                if axis_angle[0] < 0:
                    axis_angle[0] += 2
                o = np.concatenate((ee, axis_angle)).astype(np.float32)
            else:
                o = o[k]
            return o

        state_keys, action_keys = self._state_action_spec
        ret_dict    = {'states': [], 'actions': []}
        has_eef_point = 'eef_point' in traj.get(0, False)['obs']
        if has_eef_point:
            ret_dict['points'] = []
        end = len(traj)
        start = torch.randint(low=1, high=max(1, end - self._obs_T + 1 ), size=(1,))
        chosen_t = [j + start for j in range(self._obs_T)]
        if self.non_sequential:
            chosen_t = torch.randperm(end)
            chosen_t = chosen_t[chosen_t != 0][:self._obs_T]
        images = []
        images_cp = []
        for j, t in enumerate(chosen_t):
            t = t.item()
            step_t = traj.get(t)
            image = step_t['obs']['image']
            processed = self.frame_aug(task_name, image)
            images.append( processed )
            if self.aug_twice:
                images_cp.append( self.frame_aug(task_name, image, True) )
            if has_eef_point:
                ret_dict['points'].append(np.array(
                    _adjust_points(step_t['obs']['eef_point'], image.shape[:2]))[None])

            state = []
            for k in state_keys:
                state.append(_get_tensor(k, step_t))
            ret_dict['states'].append(np.concatenate(state).astype(np.float32)[None])

            if j >= 1: # TODO: just give all actions
                action = []
                for k in action_keys:
                    action.append(_get_tensor(k, step_t))
                ret_dict['actions'].append(np.concatenate(action).astype(np.float32)[None])

        for k, v in ret_dict.items():
            ret_dict[k] = np.concatenate(v, 0).astype(np.float32)

        ret_dict['images'] = torch.stack(images)
        if self.aug_twice:
            ret_dict['images_cp'] = torch.stack(images_cp)

        if self.aux_pose:
            grip_close = np.array([traj.get(i, False)['action'][-1] > 0 for i in range(1, len(traj))])
            grip_t = np.argmax(grip_close)
            drop_t = len(traj) - 1 - np.argmax(np.logical_not(grip_close)[::-1])
            aux_pose = [traj.get(t, False)['obs']['ee_aa'][:3] for t in (grip_t, drop_t)]
            ret_dict['aux_pose'] = np.concatenate(aux_pose).astype(np.float32)
        return ret_dict



class BatchMultiTaskSampler(Sampler):
    def __init__(
        self,
        task_to_idx,
        subtask_to_idx,
        sampler_spec=dict(),
        tasks_spec=dict(),
        ):
        """
        Based on the torch.BatchSampler class, add more fine-grained control over
        how to sample from different tasks and compose one batch.
        Args:
        - batch_size:
            total number of samples draw at each yield step
        - task_to_idx:
            { task_name: [all_idxs_for this task] }
        - sub_task_to_idx:
            { task_name:
                {sub_task_id: [all_idxs_for this sub-task]} }

        -> all indics in both these dict()'s should match the total dataset size,
           used for constructing samplers:
           different from the default BatchSampler,
        """
        batch_size = sampler_spec.get('batch_size', 30)
        drop_last  = sampler_spec.get('drop_last', False)
        init_ratio = sampler_spec.get('init_ratio', 'even')
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.shuffle = sampler_spec.get('shuffle', False)
        self.task_samplers = OrderedDict()
        self.task_iterators = OrderedDict() # iterators change, but samplers are fixed
        self.task_info = OrderedDict()

        for spec in tasks_spec:
            task_name  = spec.get('name', None)
            assert task_name, 'Need task name for '+str(spec)
            idxs = task_to_idx.get(task_name, None)
            assert idxs, 'Need corresponding data idxes for task '+task_name
            self.task_samplers[task_name] = OrderedDict(
                {'all_sub_tasks': SubsetRandomSampler(idxs)}) # uniformly draw from union of all sub-tasks
            self.task_iterators[task_name] = OrderedDict(
                {'all_sub_tasks': iter(SubsetRandomSampler(idxs))})

            assert task_name in subtask_to_idx.keys(), \
                'Mismatch between {} task idxs and subtasks!'.format(task_name)
            num_loaded_sub_tasks = len(subtask_to_idx[task_name].keys())
            first_id = list(subtask_to_idx[task_name].keys())[0]

            sub_task_size = len(subtask_to_idx[task_name].get(first_id))
            print("Task {:<9} loaded {} subtasks, starting from {}, should all have sizes {}".format(\
                task_name, num_loaded_sub_tasks, first_id, sub_task_size))
            for sub_task, sub_idxs in subtask_to_idx[task_name].items():
                self.task_samplers[task_name][sub_task] = SubsetRandomSampler(sub_idxs)
                assert len(sub_idxs) == sub_task_size, \
                    'Got uneven data sizes for sub-{} under the task {}!'.format(sub_task, task_name)

                self.task_iterators[task_name][sub_task] = iter(SubsetRandomSampler(sub_idxs))
            curr_task_info = {
                'size':                 len(idxs),
                'n_tasks':              len(subtask_to_idx[task_name].keys()),
                'sub_id_to_name':       {i: name for i, name in enumerate(subtask_to_idx[task_name].keys())},
                'subtask_names':        list(subtask_to_idx[task_name].keys()),
                'subtask_per_batch':    spec.get('task_per_batch', len(subtask_to_idx[task_name].keys())) # if not specified, allow all subtasks to be added to one single batch
            }
            self.task_info[task_name] = curr_task_info

        n_tasks = len(self.task_samplers.keys())
        n_total = sum([info['size'] for info in self.task_info.values()])
        actual_batch_size = 0
        for name, subtask_samplers in self.task_samplers.items():
            info = self.task_info[name]
            if init_ratio == 'even':
                sub_size = int(batch_size / n_tasks)
            elif init_ratio == 'proportional':
                sub_size = int(batch_size * info['size'] / n_total)
            else:
                raise NotImplementedError
            actual_batch_size += sub_size
            self.task_info[name]['sub_size'] = sub_size
            self.task_info[name]['sampler_len'] = int(info['size'] / sub_size) - 1 if drop_last else int(info['size'] / sub_size)
        #print(self.task_info)
        print('Finished initializing sampler. Batch sizes:')
        for k, v in self.task_info.items():
            print(k, ': per-task batch size', v['sub_size'])

        self.max_len = max([info['sampler_len'] for info in self.task_info.values()])
        print('Max length for sampler iterator:', self.max_len)
        self.n_tasks = n_tasks
        self.batch_size = actual_batch_size
        print("After distributing, actual batch size is: ", actual_batch_size)
        self.drop_last = drop_last

    def __iter__(self):
        """Given task families A,B,C, each has sub-tasks A00, A01,...
        Fix a total self.batch_size, sample different numbers of datapoints from
        each task"""
        batch = []
        for i in range(self.max_len):
            for name, info in self.task_info.items():
                subsize = info['sub_size']
                num_subtasks = info['subtask_per_batch']
                ## given N total subtasks, first select a subset of K types, then, randomly
                # select from them to construct a batch
                random.shuffle(info['subtask_names'])
                subtasks = info['subtask_names'][: num_subtasks] # get from permuted subtask names
                for idx in range(subsize):
                    random.shuffle(subtasks)
                    sub_task = subtasks[0]
                    sampler = self.task_samplers[name][sub_task]
                    iterator = self.task_iterators[name][sub_task]
                    try:
                        batch.append(next(iterator))
                    except StopIteration:
                        #print('early sstop:', i, name)
                        iterator = iter(sampler) # re-start the smaller-sized tasks
                        batch.append(next(iterator))
                        self.task_iterators[name][sub_task] = iterator

            if len(batch) == self.batch_size:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Since different task may have different data sizes,
        # define total length of sampler as number of iterations to
        # exhaust the last task
        return self.max_len

    def _update_ratio(self, new_task_sizes=dict()):
        new_batch_size = 0
        for name, new_size in new_task_sizes.items():
            assert name in self.task_info.keys(), 'Task {} is not present in this dataset'.format(name)
            self.task_info[name]['sub_size'] = new_size
            new_batch_size += new_size
        assert new_batch_size == self.batch_size, 'After updating, total batch size becomes {}, previous was {}'.format(new_batch_size, self.batch_size)

class DIYBatchSampler(Sampler):
    """
    New(0504): remove the batch_spec dict, just get those info from tasks_spec
    Use this sampler to customize any possible combination of both task families
    and sub-tasks in a batch of data.
    """
    def __init__(
        self,
        task_to_idx,
        subtask_to_idx,
        sampler_spec=dict(),
        tasks_spec=dict(),
        ):
        """
        Args:
        - batch_size:
            total number of samples draw at each yield step
        - task_to_idx: {
            task_name: [all_idxs_for this task]}
        - sub_task_to_idx: {
            task_name: {
                {sub_task_id: [all_idxs_for this sub-task]}}
           all indics in both these dict()'s should sum to the total dataset size,
        - tasks_spec:
            should additionally contain batch-constructon guide:
            explicitly specify how to contruct the batch, use this spec we should be
            able to construct a mapping from each batch index to a fixed pair
            of [task_name, subtask_id] to sample from,
            but if set shuffle=true, the sampled batch would lose this ordering,
            e.g. give a _list_: ['${place}', '${nut_hard}']
            batch spec is extracted from:
                {'place':
                        {'task_ids':     [0,1,2],
                        'n_per_task':    [5, 10, 5]}
                'nut_hard':
                        {'task_ids':     [4],
                        'n_per_task':    [6]}
                'stack':
                        {...}
                }
                will yield a batch of 36 points, where first 5 comes from pickplace subtask#0, last 6 comes from nut-assembly task#4
        - shuffle:
            if true, we lose control over how each batch is distributed to gpus
        """
        batch_size = sampler_spec.get('batch_size', 30)
        drop_last  = sampler_spec.get('drop_last', False)
        init_ratio = sampler_spec.get('init_ratio', 'even')
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.shuffle = sampler_spec.get('shuffle', False)

        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))

        self.task_samplers = OrderedDict()
        self.task_iterators = OrderedDict()
        self.task_info = OrderedDict()

        for spec in tasks_spec:
            task_name = spec.name
            idxs = task_to_idx.get(task_name)
            self.task_samplers[task_name] = OrderedDict(
                {'all_sub_tasks': SubsetRandomSampler(idxs)}) # uniformly draw from union of all sub-tasks
            self.task_iterators[task_name] = OrderedDict(
                {'all_sub_tasks': iter(SubsetRandomSampler(idxs))})
            assert task_name in subtask_to_idx.keys(), \
                'Mismatch between {} task idxs and subtasks!'.format(task_name)
            num_loaded_sub_tasks = len(subtask_to_idx[task_name].keys())
            first_id = list(subtask_to_idx[task_name].keys())[0]

            sub_task_size = len(subtask_to_idx[task_name].get(first_id))
            print("Task {} loaded {} subtasks, starting from {}, should all have sizes {}".format(\
                task_name, num_loaded_sub_tasks, first_id, sub_task_size))
            for sub_task, sub_idxs in subtask_to_idx[task_name].items():
                self.task_samplers[task_name][sub_task] = SubsetRandomSampler(sub_idxs)
                assert len(sub_idxs) == sub_task_size, \
                    'Got uneven data sizes for sub-{} under the task {}!'.format(sub_task, task_name)

                self.task_iterators[task_name][sub_task] = iter(SubsetRandomSampler(sub_idxs))
                # print('subtask indexs:', sub_task, max(sub_idxs))
            curr_task_info = {
                'size':         len(idxs),
                'n_tasks':      len(subtask_to_idx[task_name].keys()),
                'sub_id_to_name': {i: name for i, name in enumerate(subtask_to_idx[task_name].keys())},
                'traj_per_subtask': sub_task_size,
                'sampler_len':   -1 # to be decided below
            }
            self.task_info[task_name] = curr_task_info

        n_tasks = len(self.task_samplers.keys())
        n_total = sum([info['size'] for info in self.task_info.values()])

        self.idx_map = OrderedDict()
        idx = 0
        for spec in tasks_spec:
            name = spec.name
            _ids = spec.get('task_ids', None)
            n = spec.get('n_per_task', None)
            assert (_ids and n), 'Must specify which subtask ids to use and how many is contained in each batch'
            info = self.task_info[name]
            subtask_names = info.get('sub_id_to_name')
            for _id in _ids:
                subtask = subtask_names[_id]
                for _ in range(n):
                    self.idx_map[idx] = (name, subtask)
                    idx += 1
                sub_length = int(info['traj_per_subtask'] / n)
                self.task_info[name]['sampler_len'] = max(sub_length, self.task_info[name]['sampler_len'])
        #print("Index map:", self.idx_map)

        self.max_len = max([info['sampler_len'] for info in self.task_info.values()])
        print('Max length for sampler iterator:', self.max_len)
        self.n_tasks = n_tasks

        assert idx == batch_size, "The constructed batch size {} doesn't match desired {}".format(
            idx , batch_size)
        self.batch_size = idx
        self.drop_last = drop_last

        print("Shuffling to break the task ordering in each batch? ", self.shuffle)

    def __iter__(self):
        """Given task families A,B,C, each has sub-tasks A00, A01,...
        Fix a total self.batch_size, sample different numbers of datapoints from
        each task"""
        batch = []
        for i in range(self.max_len):
            for idx in range(self.batch_size):
                (name, sub_task) = self.idx_map[idx]
                # print(name, sub_task)
                sampler = self.task_samplers[name][sub_task]
                iterator = self.task_iterators[name][sub_task]
                try:
                    batch.append(next(iterator))
                except StopIteration:   #print('early sstop:', i, name)
                    iterator = iter(sampler) # re-start the smaller-sized tasks
                    batch.append(next(iterator))
                    self.task_iterators[name][sub_task] = iterator

            if len(batch) == self.batch_size:
                if self.shuffle:
                    random.shuffle(batch)
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def __len__(self):
        # Since different task may have different data sizes,
        # define total length of sampler as number of iterations to
        # exhaust the last task
        return self.max_len


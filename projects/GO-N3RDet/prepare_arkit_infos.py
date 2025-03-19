import json 
import mmcv
import numpy as np
import os
import cv2
import copy
import glob
import argparse
from pathlib import Path
from os import path as osp
from concurrent import futures as futures
import mmengine
from tqdm import tqdm  # 导入 tqdm 库用于显示进度

# 从 mmdetection3d 中导入必要的工具函数
from mmdet3d.utils import register_all_modules
from  tools.dataset_converters.update_infos_to_v2 import (
    clear_data_info_unused_keys, clear_instance_unused_keys,
    get_empty_instance, get_empty_standard_data_info)


def convert_angle_axis_to_matrix3(angle_axis):
    """Return a Matrix3 for the angle axis."""
    matrix, jacobian = cv2.Rodrigues(angle_axis)
    return matrix


def TrajStringToMatrix(traj_str):
    """Convert traj_str into translation and rotation matrices."""
    tokens = traj_str.split()
    assert len(tokens) == 7
    ts = tokens[0]
    # Rotation in angle axis
    angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
    r_w_to_p = convert_angle_axis_to_matrix3(np.asarray(angle_axis))
    # Translation
    t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
    extrinsics = np.eye(4, 4)
    extrinsics[:3, :3] = r_w_to_p
    extrinsics[:3, -1] = t_w_to_p
    Rt = np.linalg.inv(extrinsics)
    return (ts, Rt)


def st2_camera_intrinsics(filename):
    w, h, fx, fy, hw, hh = np.loadtxt(filename)
    return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


class ARKitData(object):
    """ARKitScenes数据类。"""

    def __init__(self, root_path, split='train'):
        self.root_dir = root_path
        self.split = split
        self.split_dir = os.path.join(root_path)
        self.classes = [
            "cabinet", "refrigerator", "shelf", "stove", "bed",
            "sink", "washer", "toilet", "bathtub", "oven",
            "dishwasher", "fireplace", "stool", "chair", "table",
            "tv_monitor", "sofa",
        ]
        self.cat2label = {cat: self.classes.index(cat) for cat in self.classes}
        self.label2cat = {self.cat2label[t]: t for t in self.cat2label}
        self.cat_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
        self.cat_ids2class = {
            nyu40id: i
            for i, nyu40id in enumerate(list(self.cat_ids))
        }
        assert split in ['train', 'val']
        if split == 'train':
            self.sample_id_list = [scene for scene in os.listdir(os.path.join(root_path, 'Training'))]
            self.split = 'Training'
        else:
            self.sample_id_list = [scene for scene in os.listdir(os.path.join(root_path, 'Validation'))]
            self.split = 'Validation'

    def __len__(self):
        return len(self.sample_id_list)

    def get_aligned_box_label(self, idx):
        box_file = os.path.join(self.root_dir, 'arkit_instance_data', f'{idx}_aligned_bbox.npy')
      #  mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_unaligned_box_label(self, idx):
        box_file = os.path.join(self.root_dir, 'arkit_instance_data', f'{idx}_unaligned_bbox.npy')
     #   mmcv.check_file_exist(box_file)
        return np.load(box_file)

    def get_axis_align_matrix(self, idx):
        matrix_file = os.path.join(self.root_dir, 'arkit_instance_data', f'{idx}_axis_align_matrix.npy')
      #  mmcv.check_file_exist(matrix_file)
        return np.load(matrix_file)

    def read_2d_info(self, scene):
        data_path = os.path.join(self.root_dir, self.split, scene, scene + '_frames')
        
        # 获取图像id
        depth_folder = os.path.join(data_path, "lowres_depth")
        depth_images = sorted(glob.glob(os.path.join(depth_folder, "*.png")))
        frame_ids = [os.path.basename(x) for x in depth_images]
        frame_ids = [x.split(".png")[0].split("_")[1] for x in frame_ids]
        frame_ids = [x for x in frame_ids]
        frame_ids.sort()
        
        # 读取外参
        traj_file = os.path.join(data_path, 'lowres_wide.traj')
        with open(traj_file) as f:
            self.traj = f.readlines()
        poses_from_traj = {}
        for line in self.traj:
            traj_timestamp = line.split(" ")[0]
            poses_from_traj[f"{round(float(traj_timestamp), 3):.3f}"] = TrajStringToMatrix(line)[1].tolist()

        # 获取内参
        intrinsics_from_traj = {}
        for frame_id in frame_ids:
            intrinsic_fn = os.path.join(data_path, "lowres_wide_intrinsics", f"{scene}_{frame_id}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(data_path, "lowres_wide_intrinsics",
                                            f"{scene}_{float(frame_id) - 0.001:.3f}.pincam")
            if not os.path.exists(intrinsic_fn):
                intrinsic_fn = os.path.join(data_path, "lowres_wide_intrinsics",
                                            f"{scene}_{float(frame_id) + 0.001:.3f}.pincam")
            intrinsics_from_traj[frame_id] = st2_camera_intrinsics(intrinsic_fn)
        
        image_paths = {}
        depth_paths = {}
        extrinsics = {}
        intrinsics = {}
        total_image_ids = []

        for i, vid in enumerate(frame_ids):            
            intrinsic = copy.deepcopy(intrinsics_from_traj[str(vid)]).astype(np.float32)
            if str(vid) in poses_from_traj.keys():
                frame_pose = np.array(poses_from_traj[str(vid)])
            else:
                for my_key in list(poses_from_traj.keys()):
                    if abs(float(vid) - float(my_key)) < 0.005:
                        frame_pose = np.array(poses_from_traj[str(my_key)])
            extrinsic = copy.deepcopy(frame_pose).astype(np.float32)
            img_path = os.path.join(self.split, scene, scene + '_frames', 'lowres_wide', scene + '_' + vid + '.png')
            depth_path = os.path.join(self.split, scene, scene + '_frames', 'lowres_depth', scene + '_' + vid + '.png')
            if np.all(np.isfinite(extrinsic)):
                total_image_ids.append(vid)
                image_paths[vid] = img_path
                intrinsics[vid] = intrinsic
                extrinsics[vid] = extrinsic
                depth_paths[vid] = depth_path
            else:
                print(f'invalid extrinsic for {scene}_{vid}')
        
        return total_image_ids, image_paths, depth_paths, intrinsics, extrinsics

    def get_infos(self, num_workers=4, has_label=True, sample_id_list=None):
        """获取数据信息。"""

        def process_single_scene(sample_idx):
            print(f'{self.split} sample_idx: {sample_idx}')
            tsdf_path = os.path.join(self.root_dir, 'atlas_tsdf', sample_idx)
            info_path = os.path.join(tsdf_path, 'info.json')
            with open(info_path) as f:
                info = json.load(f)

            info['split'] = self.split
            total_image_ids, image_paths, depth_paths, intrinsics, extrinsics = self.read_2d_info(sample_idx)
            info['total_image_ids'] = total_image_ids
            info['image_paths'] = image_paths
            info['depth_paths'] = depth_paths
            info['intrinsics'] = intrinsics 
            info['extrinsics'] = extrinsics
                
            if has_label:
                annotations = {}
                aligned_box_label = self.get_aligned_box_label(sample_idx)
                unaligned_box_label = self.get_unaligned_box_label(sample_idx)
                annotations['gt_num'] = aligned_box_label.shape[0]
                if annotations['gt_num'] != 0:
                    aligned_box = aligned_box_label[:, :-1]
                    unaligned_box = unaligned_box_label[:, :-1]
                    classes = aligned_box_label[:, -1]
                    annotations['name'] = np.array([
                        self.label2cat[self.cat_ids2class[classes[i]]]
                        for i in range(annotations['gt_num'])
                    ])
                    annotations['location'] = aligned_box[:, :3]
                    annotations['dimensions'] = aligned_box[:, 3:6]
                    annotations['gt_boxes_upright_depth'] = aligned_box
                    annotations['unaligned_location'] = unaligned_box[:, :3]
                    annotations['unaligned_dimensions'] = unaligned_box[:, 3:6]
                    annotations['unaligned_gt_boxes_upright_depth'] = unaligned_box
                    annotations['index'] = np.arange(annotations['gt_num'], dtype=np.int32)
                    annotations['class'] = np.array([
                        self.cat_ids2class[classes[i]]
                        for i in range(annotations['gt_num'])
                    ])
                    axis_align_matrix = self.get_axis_align_matrix(sample_idx)
                    annotations['axis_align_matrix'] = axis_align_matrix
                    info['annos'] = annotations
                else:
                    print('-' * 100)
                    print(info['split'] + '/' + info['scene'] + ' has no gt bbox, pass!')
                    print('-' * 100)
                    info = None
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        results = []
        for sample_idx in tqdm(sample_id_list, desc=f'Processing {self.split} set'):  # 使用 tqdm 显示进度
            info = process_single_scene(sample_idx)
            if info is not None:
                results.append(info)
        return results


def update_arkit_infos_nerfdet(pkl_path, out_dir):
    """将原始的 ARKit pkl 更新为 NeRF-Det 使用的新格式。"""
    METAINFO = {
        'classes': (
            "cabinet", "refrigerator", "shelf", "stove", "bed", 
            "sink", "washer", "toilet", "bathtub", "oven", 
            "dishwasher", "fireplace", "stool", "chair", "table", 
            "tv_monitor", "sofa"
        )
    }
    print(f'Reading from input file: {pkl_path}.')
    data_list = mmengine.load(pkl_path)
    print('Start updating:')
    converted_list = []
    
    ignore_class_name = set()  # 记录未找到的类名
    for ori_info_dict in tqdm(data_list, desc='Updating ARKit Infos'):
        temp_data_info = get_empty_standard_data_info()

        # 修改这里的字段名称，使其与 NeRF-Det 兼容
        temp_data_info['cam2img'] = ori_info_dict['intrinsics']
        temp_data_info['lidar2cam'] = ori_info_dict['extrinsics']
        temp_data_info['img_paths'] = ori_info_dict['image_paths']

        anns = ori_info_dict.get('annos', None)
        if anns is not None:
            temp_data_info['axis_align_matrix'] = anns['axis_align_matrix'].tolist()
            num_instances = len(anns['name'])
            instance_list = []
            for instance_id in range(num_instances):
                empty_instance = get_empty_instance()
                empty_instance['bbox_3d'] = anns['gt_boxes_upright_depth'][instance_id].tolist()

                # 检查类名是否在 METAINFO['classes'] 中
                if anns['name'][instance_id] in METAINFO['classes']:
                    empty_instance['bbox_label_3d'] = METAINFO['classes'].index(anns['name'][instance_id])
                else:
                    # 如果类名不在 METAINFO['classes'] 中，记录下来并设置为默认值 -1
                    ignore_class_name.add(anns['name'][instance_id])
                    empty_instance['bbox_label_3d'] = -1

                empty_instance = clear_instance_unused_keys(empty_instance)
                instance_list.append(empty_instance)
            temp_data_info['instances'] = instance_list
        temp_data_info, _ = clear_data_info_unused_keys(temp_data_info)
        converted_list.append(temp_data_info)
    
    pkl_name = Path(pkl_path).name
    out_path = osp.join(out_dir, pkl_name)

    # 保存为 NeRF-Det 格式的 pkl 文件
    print(f'Writing to output file: {out_path}.')
    print(f'Ignored classes: {ignore_class_name}')  # 打印未找到的类名
    metainfo = dict()
    metainfo['categories'] = {k: i for i, k in enumerate(METAINFO['classes'])}
    metainfo['dataset'] = 'arkit'
    metainfo['info_version'] = '1.1'
    
    converted_data_info = dict(metainfo=metainfo, data_list=converted_list)
    mmengine.dump(converted_data_info, out_path, 'pkl')
    print('Update completed.')


def arkit_data_prep(root_path, info_prefix, out_dir, workers):
    """准备 NeRF-Det 格式的 ARKitScenes 数据集信息文件。"""
    # 创建训练集的 pkl 文件
    arkit_dataset = ARKitData(root_path=root_path, split='train')
    # infos_train = arkit_dataset.get_infos(num_workers=workers, has_label=True)
    
    train_pkl_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    # mmengine.dump(infos_train, train_pkl_path, 'pkl')
    
    update_arkit_infos_nerfdet(pkl_path=train_pkl_path, out_dir=out_dir)

    # 创建验证集的 pkl 文件
    arkit_dataset = ARKitData(root_path=root_path, split='val')
    infos_val = arkit_dataset.get_infos(num_workers=workers, has_label=True)
    
    val_pkl_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    mmengine.dump(infos_val, val_pkl_path, 'pkl')
    
    update_arkit_infos_nerfdet(pkl_path=val_pkl_path, out_dir=out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ARKitScenes to NeRF-Det data converter')
    parser.add_argument('--root-path', type=str, default='./data/arkit', help='specify the root path of dataset')
    parser.add_argument('--out-dir', type=str, default='./data/arkit', required=False, help='name of info pkl')
    parser.add_argument('--extra-tag', type=str, default='arkit')
    parser.add_argument('--workers', type=int, default=8, help='number of threads to be used')
    args = parser.parse_args()

    register_all_modules()

    arkit_data_prep(
        root_path=args.root_path,
        info_prefix=args.extra_tag,
        out_dir=args.out_dir,
        workers=args.workers)

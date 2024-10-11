import pypose as pp
import numpy as np
import pandas
import torch
import yaml
import cv2
import os

from os import listdir
from os.path import isdir, isfile
from torch.utils.data import Dataset

from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer


def sync_data(ts_src, ts_tar):
    res = []
    j = 0
    for t in ts_tar:
        while j+1 < len(ts_src) and abs(ts_src[j+1]-t) <= abs(ts_src[j]-t):
            j += 1
        res.append(j)
    # for i in range(len(res)-1):
    #     if res[i+1] - res[i] <= 0:
    #         print('sync_data error', i, ts_tar[i:i+2], ts_src[max(0,res[i]-5):min(len(ts_src), res[i]+5)])
    return np.array(res)


def intrinsic2matrix(intrinsic):
    fx, fy, cx, cy = intrinsic
    return np.array([
        fx, 0, cx,
        0, fy, cy,
        0,  0,  1
    ], dtype=np.float32).reshape(3, 3)

def matrix2intrinsic(m):
    return np.array([m[0, 0], m[1, 1], m[0, 2], m[1, 2]], dtype=np.float32)


def stereo_rectify(left_intrinsic, left_distortion, right_intrinsic, right_distortion, width, height, right2left_pose):
    left_K = intrinsic2matrix(left_intrinsic).astype(np.float64)
    right_K = intrinsic2matrix(right_intrinsic).astype(np.float64)
    left_distortion = left_distortion.astype(np.float64)
    right_distortion = right_distortion.astype(np.float64)
    R = right2left_pose.Inv().rotation().matrix().numpy().astype(np.float64)
    T = right2left_pose.Inv().translation().numpy().astype(np.float64)

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left_K, left_distortion, right_K, right_distortion,
        (width, height), R, T, alpha=0
    )
    
    left_map = cv2.initUndistortRectifyMap(left_K, left_distortion, R1, P1, (width, height), cv2.CV_32FC1)
    right_map = cv2.initUndistortRectifyMap(right_K, right_distortion, R2, P2, (width, height), cv2.CV_32FC1)

    left_intrinsic_new = matrix2intrinsic(P1)
    right_intrinsic_new = matrix2intrinsic(P2)
    right2left_pose_new = pp.SE3([-P2[0,3]/P2[0,0],0,0, 0,0,0,1]).to(torch.float32)

    return left_intrinsic_new, right_intrinsic_new, right2left_pose_new, left_map, right_map


class TartanAirTrajFolderLoader:
    def __init__(self, datadir):

        ############################## load images ######################################################################
        imgfolder = datadir + '/image_left'
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.rgb_dts = np.ones(len(self.rgbfiles), dtype=np.float32) * 0.1
        self.rgb_ts = np.array([i for i in range(len(self.rgbfiles))], dtype=np.float64) * 0.1

        ############################## load stereo right images ######################################################################
        if isdir(datadir + '/image_right'):
            imgfolder = datadir + '/image_right'
            files = listdir(imgfolder)
            self.rgbfiles_right = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
            self.rgbfiles_right.sort()
        else:
            self.rgbfiles_right = None

        ############################## load flow ######################################################################
        if isdir(datadir + '/flow'):
            imgfolder = datadir + '/flow'
            files = listdir(imgfolder)
            self.flowfiles = [(imgfolder +'/'+ ff) for ff in files if ff.endswith('_flow.npy')]
            self.flowfiles.sort()
        else:
            self.flowfiles = None

        ############################## load depth ######################################################################
        if isdir(datadir + '/depth_left'):
            imgfolder = datadir + '/depth_left'
            files = listdir(imgfolder)
            self.depthfiles = [(imgfolder +'/'+ ff) for ff in files if ff.endswith('_depth.npy')]
            self.depthfiles.sort()
        else:
            self.depthfiles = None

        ############################## load calibrations ######################################################################
        self.intrinsic = np.array([320.0, 320.0, 320.0, 240.0], dtype=np.float32)
        self.intrinsic_right = np.array([320.0, 320.0, 320.0, 240.0], dtype=np.float32)
        self.right2left_pose = pp.SE3([0, 0.25, 0,   0, 0, 0, 1]).to(dtype=torch.float32)
        # self.right2left_pose = np.array([0, 0.25, 0,   0, 0, 0, 1], dtype=np.float32)
        self.require_undistort = False

        ############################## load gt poses ######################################################################
        posefile = datadir + '/pose_left.txt'
        self.poses = np.loadtxt(posefile).astype(np.float32)
        self.vels = None

        ############################## load imu data ######################################################################
        if isdir(datadir + '/imu'):
            self.imu_dts = np.ones(len(self.rgbfiles)*10, dtype=np.float32) * 0.01
            self.imu_ts = np.array([i for i in range(len(self.rgbfiles)*10)], dtype=np.float64) * 0.01
            self.rgb2imu_sync = np.array([i for i in range(len(self.rgbfiles))]) * 10
            self.rgb2imu_pose = pp.SE3([0, 0, 0,   0, 0, 0, 1]).to(dtype=torch.float32)
            self.gravity = 0

            imudir = datadir + '/imu'
            # acceleration in the body frame
            self.accels = np.load(imudir + '/acc_nograv_body.npy')
            # angular rate in the body frame
            self.gyros = np.load(imudir + '/gyro.npy')
            # velocity in the world frame
            self.vels = np.load(imudir + '/vel_global.npy')

            with open(imudir + '/parameter.yaml', 'r') as file:
                paras = yaml.safe_load(file)
            self.accel_bias = np.array(paras['acc_zero_bias'], dtype=np.float32)
            self.gyro_bias = np.array(paras['gyro_zero_bias'], dtype=np.float32)

            self.has_imu = True


class EuRoCTrajFolderLoader:
    def __init__(self, datadir):
        all_timestamps = []

        ############################## load images ######################################################################
        df = pandas.read_csv(datadir + '/cam0/data.csv')
        timestamps_left = df.values[:, 0].astype(int) // int(1e6)
        all_timestamps.append(timestamps_left)
        self.rgbfiles = datadir + '/cam0/data/' + df.values[:, 1]

        ############################## load stereo right images ######################################################################
        if isfile(datadir + '/cam1/data.csv'):
            df = pandas.read_csv(datadir + '/cam1/data.csv')
            timestamps_right = df.values[:, 0].astype(int) // int(1e6)
            all_timestamps.append(timestamps_right)
            self.rgbfiles_right = datadir + '/cam1/data/' + df.values[:, 1]
        else:
            self.rgbfiles_right = None

        ############################## load calibrations ######################################################################
        with open(datadir + '/cam0/sensor.yaml') as f:
            res = yaml.load(f.read(), Loader=yaml.FullLoader)
            self.intrinsic = np.array(res['intrinsics'], dtype=np.float32)
            distortion = np.array(res['distortion_coefficients'], dtype=np.float32)
            T_BL = np.array(res['T_BS']['data'], dtype=np.float32).reshape(4, 4)
        
        if self.rgbfiles_right is not None:
            with open(datadir + '/cam1/sensor.yaml') as f:
                res = yaml.load(f.read(), Loader=yaml.FullLoader)
                self.intrinsic_right = np.array(res['intrinsics'], dtype=np.float32)
                distortion_right = np.array(res['distortion_coefficients'], dtype=np.float32)
                T_BR = np.array(res['T_BS']['data'], dtype=np.float32).reshape(4, 4)

        if self.rgbfiles_right is not None:
            T_LR = np.matmul(np.linalg.inv(T_BL), T_BR)
            self.right2left_pose = pp.from_matrix(torch.tensor(T_LR), ltype=pp.SE3_type).to(dtype=torch.float32)

            self.require_undistort = True
            img = cv2.imread(self.rgbfiles_right[0])
            h, w = img.shape[:2]
            self.intrinsic, self.intrinsic_right, self.right2left_pose, self.imgmap, self.imgmap_right=stereo_rectify(
                self.intrinsic, distortion, self.intrinsic_right, distortion_right, w, h, self.right2left_pose)
        else:
            self.require_undistort = False

        ############################## load gt poses ######################################################################
        df = pandas.read_csv(datadir + '/state_groundtruth_estimate0/data.csv')
        timestamps_pose = df.values[:, 0].astype(int) // int(1e6)   # ms to s
        all_timestamps.append(timestamps_pose)
        self.poses = df.values[:, (1,2,3, 5,6,7,4)].astype(np.float32)  # x, y, z, qx, qy, qz, qw (right format of TUM dataset)
        self.vels = df.values[:, 8:11].astype(np.float32)
        accel_bias = df.values[:, 14:17].astype(np.float32)
        gyro_bias = df.values[:, 11:14].astype(np.float32)

        ############################## align timestamps ######################################################################
        timestamps = set(all_timestamps[0])
        for i in range(1, len(all_timestamps)):
            timestamps = timestamps.intersection(set(all_timestamps[i]))
        self.rgbfiles = self.rgbfiles[[i for i, t in enumerate(timestamps_left) if t in timestamps]]
        if self.rgbfiles_right is not None:
            self.rgbfiles_right = self.rgbfiles_right[[i for i, t in enumerate(timestamps_right) if t in timestamps]]
        self.poses = self.poses[[i for i, t in enumerate(timestamps_pose) if t in timestamps]]
        self.vels = self.vels[[i for i, t in enumerate(timestamps_pose) if t in timestamps]]
        timestamps = np.array(list(timestamps))
        timestamps.sort()
        self.rgb_dts = np.diff(timestamps).astype(np.float32) * 1e-3
        self.rgb_ts = np.array(timestamps).astype(np.float64) * 1e-3

        ############################## load imu data ######################################################################
        if isfile(datadir + '/imu0/data.csv'):
            df = pandas.read_csv(datadir + '/imu0/data.csv')
            timestamps_imu = df.values[:, 0].astype(int) // int(1e6)
            accels = df.values[:, 4:7].astype(np.float32)
            gyros = df.values[:, 1:4].astype(np.float32)

            imu2pose_sync = sync_data(timestamps_pose, timestamps_imu)
            # self.accels = accels - accel_bias[imu2pose_sync]
            # self.gyros = gyros - gyro_bias[imu2pose_sync]
            self.accels = accels
            self.gyros = gyros
            self.accel_bias = np.mean(accel_bias[imu2pose_sync], axis=0)
            self.gyro_bias = np.mean(gyro_bias[imu2pose_sync], axis=0)

            self.imu_dts = np.diff(timestamps_imu).astype(np.float32) * 1e-3
            self.imu_ts = np.array(timestamps_imu).astype(np.float64) * 1e-3
            
            self.rgb2imu_sync = sync_data(timestamps_imu, timestamps)

            with open(datadir + '/imu0/sensor.yaml') as f:
                res = yaml.load(f.read(), Loader=yaml.FullLoader)
                T_BI = np.array(res['T_BS']['data'], dtype=np.float32).reshape(4, 4)
                T_IL = np.matmul(np.linalg.inv(T_BI), T_BL)
                self.rgb2imu_pose = pp.from_matrix(torch.tensor(T_IL), ltype=pp.SE3_type).to(dtype=torch.float32)

            self.gravity = 9.81

            self.has_imu = True

        else:
            self.has_imu = False


class KITTITrajFolderLoader:
    def __init__(self, datadir):
        import pykitti

        datadir_split = datadir.split('/')
        # Change this to the directory where you store KITTI data
        basedir = '/'.join(datadir_split[:-2])

        # Specify the dataset to load
        date = datadir_split[-2]
        drive = datadir_split[-1].split('_')[-2]

        # Load the data. Optionally, specify the frame range to load.
        dataset = pykitti.raw(basedir, date, drive)
        # dataset = pykitti.raw(basedir, date, drive, frames=range(0, 20, 5))

        # dataset.calib:         Calibration data are accessible as a named tuple
        # dataset.timestamps:    Timestamps are parsed into a list of datetime objects
        # dataset.oxts:          List of OXTS packets and 6-dof poses as named tuples
        # dataset.camN:          Returns a generator that loads individual images from camera N
        # dataset.get_camN(idx): Returns the image from camera N at idx
        # dataset.gray:          Returns a generator that loads monochrome stereo pairs (cam0, cam1)
        # dataset.get_gray(idx): Returns the monochrome stereo pair at idx
        # dataset.rgb:           Returns a generator that loads RGB stereo pairs (cam2, cam3)
        # dataset.get_rgb(idx):  Returns the RGB stereo pair at idx
        # dataset.velo:          Returns a generator that loads velodyne scans as [x,y,z,reflectance]
        # dataset.get_velo(idx): Returns the velodyne scan at idx

        ############################## load times ######################################################################
        # timestamps = np.array([t.timestamp() for t in dataset.timestamps])
        ts_imu = self.load_timestamps(datadir, 'oxts')
        ts_rgb = self.load_timestamps(datadir, 'image_02')
        # self.rgb2imu_sync = np.array([i for i in range(len(self.rgbfiles))])
        self.rgb2imu_sync = sync_data(ts_imu, ts_rgb)

        ############################## load images ######################################################################
        self.rgbfiles = dataset.cam2_files
        self.rgb_dts = np.diff(ts_rgb).astype(np.float32)
        self.rgb_ts = np.array(ts_rgb).astype(np.float64) - ts_rgb[0]

        ############################## load stereo right images ######################################################################
        self.rgbfiles_right = dataset.cam3_files

        ############################## load calibrations ######################################################################
        K = dataset.calib.K_cam2
        self.intrinsic = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
        K = dataset.calib.K_cam3
        self.intrinsic_right = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])

        T_LI = dataset.calib.T_cam2_imu
        T_RI = dataset.calib.T_cam3_imu
        T_LR = np.matmul(T_LI, np.linalg.inv(T_RI))
        self.right2left_pose = pp.from_matrix(torch.tensor(T_LR), ltype=pp.SE3_type).to(dtype=torch.float32)

        self.require_undistort = False

        ############################## load gt poses ######################################################################
        T_w_imu = np.array([oxts_frame.T_w_imu for oxts_frame in dataset.oxts])
        T_w_imu = T_w_imu[self.rgb2imu_sync]
        self.poses = pp.from_matrix(torch.tensor(T_w_imu), ltype=pp.SE3_type).to(dtype=torch.float32)
        
        vels_local = torch.tensor([[oxts_frame.packet.vf, oxts_frame.packet.vl, oxts_frame.packet.vu] for oxts_frame in dataset.oxts], dtype=torch.float32)
        vels_local = vels_local[self.rgb2imu_sync]
        self.vels = self.poses.rotation() @ vels_local

        self.poses = self.poses.numpy()
        self.vels = self.vels.numpy()

        ############################## load imu data ######################################################################
        self.accels = np.array([[oxts_frame.packet.ax, oxts_frame.packet.ay, oxts_frame.packet.az] for oxts_frame in dataset.oxts]).astype(np.float32)
        self.gyros = np.array([[oxts_frame.packet.wx, oxts_frame.packet.wy, oxts_frame.packet.wz] for oxts_frame in dataset.oxts]).astype(np.float32)

        self.accel_bias = np.zeros(3, dtype=np.float32)
        self.gyro_bias = np.zeros(3, dtype=np.float32)

        self.imu_dts = np.diff(ts_imu).astype(np.float32)
        self.imu_ts = np.array(ts_imu).astype(np.float64) - ts_imu[0]

        T_IL = np.linalg.inv(T_LI)
        self.rgb2imu_pose = pp.from_matrix(torch.tensor(T_IL), ltype=pp.SE3_type).to(dtype=torch.float32)

        self.gravity = 9.81

        self.has_imu = True

    def load_timestamps(self, datapath, subfolder):
        import datetime as dt

        """Load timestamps from file."""
        timestamp_file = os.path.join(
            datapath, subfolder, 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits.
                t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t.timestamp())
        timestamps.sort()

        return timestamps


class TrajFolderDatasetBase(Dataset):
    def __init__(self, datadir, datatype, transform=None, start_frame=0, end_frame=-1, loader=None):
        if loader is None:
            if datatype == 'tartanair':
                loader = TartanAirTrajFolderLoader(datadir)
            elif datatype == 'euroc':
                loader = EuRoCTrajFolderLoader(datadir)
            elif datatype == 'kitti':
                loader = KITTITrajFolderLoader(datadir)

        if end_frame <= 0:
            end_frame += len(loader.rgbfiles)

        self.datadir = datadir
        self.datatype = datatype
        self.transform = transform
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.loader = loader
        
        self.rgbfiles = loader.rgbfiles[start_frame:end_frame]
        self.rgb_dts = loader.rgb_dts[start_frame:end_frame-1]
        self.rgb_ts = loader.rgb_ts[start_frame:end_frame]
        self.num_img = len(self.rgbfiles)

        try:
            self.rgbfiles_right = loader.rgbfiles_right[start_frame:end_frame]
        except:
            self.rgbfiles_right = None

        try:
            self.flowfiles = loader.flowfiles[start_frame:end_frame-1]
        except:
            self.flowfiles = None

        try:
            self.depthfiles = loader.depthfiles[start_frame:end_frame]
        except:
            self.depthfiles = None

        self.intrinsic = loader.intrinsic
        try:
            self.intrinsic_right = loader.intrinsic_right
            self.right2left_pose = loader.right2left_pose
        except:
            pass

        self.poses = loader.poses[start_frame:end_frame]

        try:
            self.vels = loader.vels[start_frame:end_frame]
        except:
            self.vels = None

        if loader.has_imu:
            self.rgb2imu_sync = loader.rgb2imu_sync[start_frame:end_frame]
            start_imu = self.rgb2imu_sync[0]
            end_imu = self.rgb2imu_sync[-1] + 1
            self.rgb2imu_sync -= start_imu

            self.accels = loader.accels[start_imu:end_imu]
            self.gyros = loader.gyros[start_imu:end_imu]
            self.imu_dts = loader.imu_dts[start_imu:end_imu-1]
            self.imu_ts = loader.imu_ts[start_imu:end_imu]
            
            self.rgb2imu_pose = loader.rgb2imu_pose
            self.imu_init = {'rot':self.poses[0, 3:], 'pos':self.poses[0, :3], 'vel':self.vels[0]}
            self.gravity = loader.gravity

            self.accel_bias = loader.accel_bias
            self.gyro_bias = loader.gyro_bias

            self.imu_motions = None
            self.has_imu = True

        else:
            self.has_imu = False

        if loader.require_undistort:
            self.imgmap = loader.imgmap
            try:
                self.imgmap_right = loader.imgmap_right
            except:
                pass
            self.require_undistort = True
        else:
            self.require_undistort = False

        self.links = None
        self.num_link = 0

        del loader


class TrajFolderDataset(TrajFolderDatasetBase):
    def __init__(self, datadir, datatype, transform=None, start_frame=0, end_frame=-1, loader=None, links=None):
        super(TrajFolderDataset, self).__init__(datadir, datatype, transform, start_frame, end_frame, loader)

        if links is None:
            self.links = [[i, i+1] for i in range(self.num_img-1)]
        else:
            self.links = links
        self.num_link = len(self.links)

        self.motions = self.calc_motions_by_links(self.links)

    def __getitem__(self, idx):
        return self.get_pair(self.links[idx][0], self.links[idx][1])
    
    def __len__(self):
        return self.num_link

    def calc_motions_by_links(self, links):
        if self.poses is None:
            return None

        SEs = pos_quats2SEs(self.poses)
        matrix = pose2motion(SEs, links=links)
        motions = SEs2ses(matrix).astype(np.float32)
        return motions
    
    def undistort(self, img, is_right=False):
        if not self.require_undistort:
            return img
        imgmap = self.imgmap_right if is_right else self.imgmap
        dst = cv2.remap(img, imgmap[0], imgmap[1], cv2.INTER_AREA)
        return dst

    def get_pair(self, i, j):
        res = {}

        img0 = cv2.imread(self.rgbfiles[i], cv2.IMREAD_COLOR)
        img1 = cv2.imread(self.rgbfiles[j], cv2.IMREAD_COLOR)
        img0 = self.undistort(img0)
        img1 = self.undistort(img1)
        res['img0'] = [img0]
        res['img1'] = [img1]
        # load data of cam2
        if self.rgbfiles_right is not None:
            img0_r = cv2.imread(self.rgbfiles_right[i], cv2.IMREAD_COLOR)
            img1_r = cv2.imread(self.rgbfiles_right[j], cv2.IMREAD_COLOR)
            img0_r = self.undistort(img0_r, True)
            img1_r = self.undistort(img1_r, True)
            res['img0_r'] = [img0_r]
            res['img1_r'] = [img1_r]

        h, w, _ = img0.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.intrinsic[0], self.intrinsic[1], self.intrinsic[2], self.intrinsic[3])
        res['intrinsic'] = [intrinsicLayer]

        res['intrinsic_calib'] = self.intrinsic.copy()

        if self.transform:
            res = self.transform(res)

        res['link'] = np.array([i, j])

        res['dt'] = np.sum(self.rgb_dts[min(i, j):max(i, j)])

        res['datatype'] = self.datatype

        res['motion'] = (pp.SE3(self.poses[i]).Inv() @ pp.SE3(self.poses[j])).numpy()

        if self.right2left_pose != None:
            res['extrinsic'] = self.right2left_pose.clone().numpy()

        res['img0_file'] = self.rgbfiles[i]
        res['img1_file'] = self.rgbfiles[j]
        res['img0_r_file'] = self.rgbfiles_right[i]
        res['img1_r_file'] = self.rgbfiles_right[j]

        return res
    
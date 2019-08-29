import tensorflow as tf
from collections import OrderedDict
from pdb import set_trace as st
from dovebirdia.datasets.unpa_op import UNPAOPDataset

params = dict()
params['dataset_base_dir'] = '/home/mlweiss/Documents/wpi/research/data/pose/head_pose/Head_Pose_Database_UPNA/openpose/'
params['user'] = '01'
params['video'] = '01'
params['landmarks'] = OrderedDict([
    # left eyebrow
    ('LEB-O',(1,17)),
    ('LEB-OM',(2,18)),
    ('LEB-M',(3,19)),
    ('LEB-IM',(4,20)),
    ('LEB-I',(5,21)),
    # right eyebrow
    ('REB-O',(6,26)),
    ('REB-OM',(7,25)),
    ('REB-M',(8,24)),
    ('REB-IM',(9,23)),
    ('REB-I',(10,22)),
    # left eye
    ('LE-O',(11,36)),
    ('LE-I',(15,39)),
    # right eye
    ('RE-O',(19,45)),
    ('RE-I',(23,42)),
    # mouth
    ('M-L',(38,48)),
    ('M-TLM',(39,50)),
    ('M-TM',(40,51)),
    ('M-TRM',(41,52)),
    ('M-R',(42,54)),
    ('M-BRM',(43,56)),
    ('M-BM',(44,57)),
    ('M-TLM',(45,58)),
])

dataset = UNPAOPDataset(params)
ds = dataset.getDataset()
st()

from yacs.config import CfgNode as CN

_C = CN()

_C.DATA = CN()
_C.DATA.train_root = "/mnt/nas/chenjiayang/Projects/RFMfold/data/bpRNA/TR0"
_C.DATA.val_root = "/mnt/nas/chenjiayang/Projects/RFMfold/data/bpRNA/TS0"
_C.DATA.energy_dict_path = "./bp_fea/avg_energy_stacking_k2.pkl"
_C.DATA.energy_dist_dict_path = "./bp_fea/avg_energy_dist_k2.pkl"
_C.DATA.feature_parent_dir = CN()
_C.DATA.feature_parent_dir.train = "/mnt/nas/chenjiayang/Projects/RFMfold/ss_fea/bpRNA/TR0"
_C.DATA.feature_parent_dir.val = "/mnt/nas/chenjiayang/Projects/RFMfold/ss_fea/bpRNA/TS0"
_C.DATA.active_methods = ("mxfold2", "rinalmo", "rnaformer") # None ["mxfold2", "rinalmo", "rnafm", "rnaformer"]

_C.MODEL = CN()
_C.MODEL.main_ch = 16
_C.MODEL.energy_ch = 2
_C.MODEL.ss_fea_ch = -1  # ss_fea_ch is set dynamically
_C.MODEL.base_ch = 128
_C.MODEL.depth = 6
_C.MODEL.drop_p = 0.15
_C.MODEL.dilations = (1, 2, 4, 8, 16)

_C.OPTIMIZER = CN()
_C.OPTIMIZER.lr = 1e-4

_C.SCHEDULER = CN()
_C.SCHEDULER.step_size = 10
_C.SCHEDULER.gamma = 0.5

_C.SAMPLER = CN()
_C.SAMPLER.start_ratio = 0.5
_C.SAMPLER.end_ratio = 0.6
_C.SAMPLER.step_factor = 0.01
_C.SAMPLER.step_every_n_epochs = 2

_C.LOADER = CN()
_C.LOADER.train = CN()
_C.LOADER.train.batch_size = 8
_C.LOADER.train.shuffle = True
_C.LOADER.train.num_workers = 4
_C.LOADER.val = CN()
_C.LOADER.val.batch_size = 8
_C.LOADER.val.shuffle = False
_C.LOADER.val.num_workers = 4

_C.TRAINER = CN()
_C.TRAINER.max_epochs = 100
_C.TRAINER.accelerator = "gpu"
_C.TRAINER.devices = "auto"
_C.TRAINER.strategy = "ddp"
_C.TRAINER.precision = "16-mixed"
_C.TRAINER.log_every_n_steps = 10
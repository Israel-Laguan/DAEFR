import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor, TQDMProgressBar
from pytorch_lightning.utilities import rank_zero_only
import random
import pdb

# Enable cudnn.benchmark for optimized convolution algorithms
# DAEFR uses fixed 512x512 input size - perfect for this optimization
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("Enabled cudnn.benchmark for optimized convolution algorithms")

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "--pretrain",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="pretrain with existed weights",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "--random-seed",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )

    parser.add_argument(
        "--root-path",
        type=str,
        default="./",
        help="root path for saving checkpoints and logs"
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="number of gpu nodes",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPU IDs to use (e.g., '0' or '0,1'). Required for training.",
    )

    return parser


def nondefault_trainer_args(opt):
    """In PL 2.x, we manually track which args differ from defaults."""
    # Default trainer args that we care about
    default_args = {
        "max_epochs": None,
        "gpus": None,
        "num_nodes": 1,
        "accumulate_grad_batches": 1,
    }
    return sorted(k for k in default_args if hasattr(opt, k) and getattr(opt, k) != default_args[k])


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    if 'basicsr.data' in config["target"] or 'FFHQ' in config["target"]:
        return get_obj_from_str(config["target"])(config.get("params", dict()))
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, pin_memory=False, persistent_workers=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # import pdb
            # pdb.set_trace()
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            print("Project config")
            print(self.config.pretty())
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(self.lightning_config.pretty())
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))


class ImageLogger(Callback):
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            pl.loggers.TensorBoardLogger: self._tensorboard,
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _tensorboard(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)

            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy()
            grid = (grid*255).astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if (self.check_frequency(batch_idx) and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            with torch.no_grad():
                images = pl_module.log_images(batch, split=split)

            for k in images:
                N = min(images[k].shape[0], self.max_images)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.log_img(pl_module, batch, batch_idx, split="val")



if __name__ == "__main__":
    # custom parser to specify config files, train, test and debug mode,
    # postfix, resume.
    # `--key value` arguments are interpreted as arguments to the trainer.
    # `nested.key=value` arguments are interpreted as config parameters.
    # configs are merged from left-to-right followed by command line parameters.

    # model:
    #   base_learning_rate: float
    #   target: path to lightning module
    #   params:
    #       key: value
    # data:
    #   target: main.DataModuleFromConfig
    #   params:
    #      batch_size: int
    #      wrap: bool
    #      train:
    #          target: path to train dataset
    #          params:
    #              key: value
    #      validation:
    #          target: path to validation dataset
    #          params:
    #              key: value
    #      test:
    #          target: path to test dataset
    #          params:
    #              key: value
    # lightning: (optional, has sane defaults and can be specified on cmdline)
    #   trainer:
    #       additional arguments to trainer
    #   logger:
    #       logger to instantiate
    #   modelcheckpoint:
    #       modelcheckpoint to instantiate
    #   callbacks:
    #       callback1:
    #           target: importpath
    #           params:
    #               key: value
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # pdb.set_trace()

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    # GPU guard: training requires GPU
    if opt.train and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available but training was requested. Training on CPU is not supported.")
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    # Resume logic: handle both standalone checkpoints and logs-based checkpoints
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        
        opt.resume_from_checkpoint = opt.resume
        
        if os.path.isdir(opt.resume):
            # Resume from a logs directory - find last checkpoint and load configs
            logdir = opt.resume.rstrip("/")
            opt.resume_from_checkpoint = os.path.join(logdir, "checkpoints", "last.ckpt")
            # Auto-load base configs from this run
            base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
            opt.base = base_configs + opt.base
            # Extract run name from logs directory
            _tmp = logdir.split("/")
            if "logs" in _tmp:
                nowname = _tmp[_tmp.index("logs")+1] + opt.postfix
            else:
                nowname = os.path.basename(logdir) + opt.postfix
        else:
            # Standalone checkpoint file - use CLI --base for configs
            # Create new run name based on config or timestamp
            if opt.name:
                name = "_"+opt.name
            elif opt.base:
                cfg_fname = os.path.split(opt.base[0])[-1]
                cfg_name = os.path.splitext(cfg_fname)[0]
                name = "_"+cfg_name
            else:
                name = ""
            nowname = now + name + opt.postfix
            logdir = os.path.join(opt.root_path, "logs", nowname)
    else:
        # Fresh run
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""
        nowname = now+name+opt.postfix
        logdir = os.path.join(opt.root_path, "logs", nowname)

    # Flat structure: checkpoints in experiments/, logs in experiments/logs/
    ckptdir = os.path.join(opt.root_path, nowname)
    cfgdir = os.path.join(logdir, "configs")

    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # PL 2.x: use strategy instead of accelerator for ddp
        # Enable find_unused_parameters for DAEFR model with some unused params
        trainer_config["strategy"] = "ddp_find_unused_parameters_true"
        # Copy CLI args to trainer_config
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        # Handle GPU/CPU setup for PL 2.x
        if "gpus" in trainer_config and trainer_config["gpus"]:
            # PL 2.x: use accelerator="gpu" and devices instead of gpus
            gpuinfo = trainer_config.pop("gpus")
            trainer_config["accelerator"] = "gpu"
            trainer_config["devices"] = gpuinfo if isinstance(gpuinfo, int) else len(gpuinfo.strip(",").split(","))
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        else:
            trainer_config["accelerator"] = "cpu"
            cpu = True
        trainer_config["max_epochs"] = config.model.max_epochs
        trainer_config["profiler"] = "simple"
        lightning_config.trainer = trainer_config
        if opt.resume:
            print('====== Resume from last checkpoint ======')
            print(f'====== Loading checkpoint from {opt.resume} ======')
            print('====== Skipping pretrained weights (ckpt_path_HQ and ckpt_path_LQ) ======')
            config.model.params.ckpt_path_HQ = None
            config.model.params.ckpt_path_LQ = None
        # model
        model = instantiate_from_config(config.model)

        # torch.compile is now handled inside the LightningModule setup() method

        # trainer and callbacks
        trainer_kwargs = dict()
        # trainer_kwargs['sync_batchnorm'] = True
        
        # default logger configs
        # NOTE wandb < 0.10.0 interferes with shutdown
        # wandb >= 0.10.0 seems to fix it but still interferes with pudb
        # debugging (wrongly sized pudb ui)
        # thus prefer testtube for now
        default_logger_cfgs = {
            "csv": {
                "target": "pytorch_lightning.loggers.CSVLogger",
                "params": {
                    "save_dir": logdir,
                    "name": "csv",
                }
            },
        }
        default_logger_cfg = default_logger_cfgs["csv"]
        logger_cfg = lightning_config.get("logger") or OmegaConf.create()
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                # "filename": "{epoch:06}",
                "filename": "{epoch:06}-{Rec_loss}-{BCE_loss}-{L2_loss}",
                "monitor": "BCE_loss",
                "verbose": True,
                "save_last": True,
                "save_top_k": 3,
                "every_n_epochs": 1
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3
        # pdb.set_trace()
        modelckpt_cfg = lightning_config.get("modelcheckpoint") or OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        # PL 2.x: checkpoint_callback removed, add to callbacks list instead
        checkpoint_callback = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main_DAEFR.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main_DAEFR.ImageLogger",
                "params": {
                    "batch_frequency": 750,
                    "max_images": 4,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main_DAEFR.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
        }
        callbacks_cfg = lightning_config.get("callbacks") or OmegaConf.create()
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
        # Add checkpoint callback to the list (PL 2.x requirement)
        trainer_kwargs["callbacks"].append(checkpoint_callback)
        # Add TQDMProgressBar with custom refresh rate to reduce terminal spam
        trainer_kwargs["callbacks"].append(TQDMProgressBar(refresh_rate=10))

        # PL 2.x: direct Trainer instantiation instead of from_argparse_args
        trainer = Trainer(**trainer_config, **trainer_kwargs)

        # data
        data = instantiate_from_config(config.data)
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        data.prepare_data()
        data.setup()

        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        if not cpu:
            ngpu = len(lightning_config.trainer.devices) if isinstance(lightning_config.trainer.devices, list) else lightning_config.trainer.devices
        else:
            ngpu = 1
        accumulate_grad_batches = lightning_config.trainer.get("accumulate_grad_batches", 1)
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * trainer.num_nodes * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (num_nodes) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, trainer.num_nodes, bs, base_lr))

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        # run
        if opt.train:
            try:
                trainer.fit(model, data, ckpt_path=opt.resume_from_checkpoint if opt.resume else None, weights_only=False)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)

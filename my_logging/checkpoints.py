from pathlib import Path
import os, glob
import torch

from my_logging.loggers import Logger


class Errors(Logger):
    ERROR_FILE = "errors.csv"
    ERRORS_HEADER_FORMAT = (
        "{idx},{epoch},{tl},{vl},{ta},{va},{qta},{qva},{dist},{sp},{compl},{s_compl}\n"
    )
    ERRORS_FORMAT = "{idx},{epoch},{tl:.4f},{vl:.4f},{ta:.4f},{va:.4f},{qta:.4f},{qva:.4f},{dist:.4f},{sp:.4f},{compl},{s_compl}\n"

    def __init__(self, checkpoints, **kwargs):
        super().__init__(**kwargs)
        self.cp = checkpoints
        self.errors_path = self.cp.path / self.ERROR_FILE

        # prepare errors file
        with open(self.errors_path, "w") as f:
            new_var = self.ERRORS_HEADER_FORMAT.format(
                idx="idx",
                epoch="epoch",
                tl="train_loss",
                vl="valid_loss",
                ta="train_acc",
                va="valid_acc",
                qta="quantized_train_acc",
                qva="quantized_valid_acc",
                dist="distance",
                sp="sparsity",
                compl="complexity",
                s_compl="simple_complexity",
            )
            f.write(new_var)

    def epoch_end(
        self,
        train_loss,
        valid_loss,
        train_acc,
        valid_acc,
        q_train_acc,
        q_valid_acc,
        distance,
        sparsity,
        compl,
        simple_compl,
        **kwargs,
    ):
        # log errors
        with open(self.errors_path, "a") as f:
            f.write(
                self.ERRORS_FORMAT.format(
                    idx=self.cp.t.conf_idx,
                    epoch=self.cp.t.epoch,
                    tl=train_loss,
                    vl=valid_loss,
                    ta=train_acc,
                    va=valid_acc,
                    qta=q_train_acc,
                    qva=q_valid_acc,
                    dist=distance,
                    sp=sparsity,
                    compl=compl,
                    s_compl=simple_compl,
                )
            )


class Configs(Logger):
    CONFIGS_FILE = "configs.csv"
    CONFIGS_FORMAT = "{seed},{lr},{batch_size},{ternary},{a},{b}\n"

    def __init__(self, checkpoints, **kwargs):
        super().__init__(**kwargs)
        self.cp = checkpoints
        self.configs_path = self.cp.path / self.CONFIGS_FILE

        # prepare configs file
        with open(self.configs_path, "w") as f:
            f.write(
                self.CONFIGS_FORMAT.format(
                    seed="seed",
                    lr="learning_rate",
                    batch_size="batch_size",
                    ternary="ternary",
                    a="a",
                    b="b",
                )
            )

    def training_start(self, **kwargs):
        conf = self.cp.t.conf
        # log the configuration
        with open(self.configs_path, "a") as f:
            f.write(
                self.CONFIGS_FORMAT.format(
                    seed=conf.seed,
                    lr=conf.lr,
                    batch_size=conf.batch_size,
                    ternary=conf.ternary,
                    a=conf.a,
                    b=conf.b,
                )
            )


class Models(Logger):
    """
    Makes a checkpoint of the trained models every couple epochs
    """

    MODEL_FILE = "config{idx:02d}_epoch{epoch:03d}.pth"

    def __init__(self, checkpoints, **kwargs):
        super().__init__(**kwargs)
        self.cp = checkpoints
        self.model_path = self.cp.path / self.MODEL_FILE

    def epoch_end(self, **kwargs):
        self.save_model()

    def save_model(self):
        curr_model_path = str(self.model_path).format(
            idx=self.cp.t.conf_idx, epoch=self.cp.t.epoch
        )
        try:
            torch.save(self.cp.t.model, curr_model_path)
        except Exception as inst:
            print(f"Could not save model to {curr_model_path}: {inst}")


class BestModels(Models):
    """
    Keeps track of/saves the best models w.r.t. to a metric.
    """

    MODEL_DIR = "best"
    MODEL_FILE = "config{idx:02d}_epoch{epoch:03d}.pth"

    def __init__(self, checkpoints, **kwargs):
        super().__init__(checkpoints, **kwargs)
        self.model_path = self.cp.path / self.MODEL_DIR / self.MODEL_FILE
        self.stats = []
        if not os.path.isdir(self.model_path.parent):
            os.mkdir(self.model_path.parent)

    def is_optimal(self, stats: list[tuple[float, float]]) -> bool:
        sorted_list = sorted(stats)
        assert all(i[0] < j[0] for i, j in zip(sorted_list[:-1], sorted_list[1:]))
        return all(i[1] > j[1] for i, j in zip(sorted_list[:-1], sorted_list[1:]))

    def insert(self, tup: tuple[float, float], stats: list[tuple[float, float]]):
        assert self.is_optimal(self.stats)

        to_delete_indices = []
        for idx, tup2 in enumerate(stats):
            if tup[0] <= tup2[0] and tup[1] <= tup2[1]:
                return
            if tup[0] >= tup2[0] and tup[1] >= tup2[1]:
                to_delete_indices.append(idx)

        for idx in sorted(to_delete_indices, reverse=True):
            # delete the correct model from the directory
            tup2 = stats[idx]
            to_delete_path = str(self.model_path).format(idx=tup2[2], epoch=tup2[3])
            print(f"Deleting {to_delete_path}...")
            os.remove(to_delete_path)
            del self.stats[idx]

        # add the model to the directory
        to_save_path = str(self.model_path).format(idx=tup[0], epoch=tup[1])
        print(f"Saving into {to_save_path}...")
        torch.save(self.cp.t.model, to_save_path)
        self.stats.append(tup)

    def epoch_end(self, q_valid_acc, simple_compl, **kwargs):
        # self.insert((q_valid_acc, -simple_compl, self.cp.t.conf_idx, self.cp.t.epoch), self.stats)
        pass

    def training_end(self):
        print(self.stats)
        print()


class Checkpoints(Logger):
    def __init__(self, path: Path, error_every=1, config_every=1, model_every=1):
        super().__init__(log_every=1)
        self.path = Path(path)
        assert path is not None

        self.path.mkdir(exist_ok=True, parents=True)
        for f in glob.glob(self.path.__str__() + "/**", recursive=True):
            print(f"Deleting {f}...")
            if not os.path.isdir(f):
                os.remove(f)
        self.error_logger = Errors(self, log_every=error_every)
        self.config_logger = Configs(self, log_every=config_every)
        self.model_logger = Models(self, log_every=model_every)
        self.best_model_logger = BestModels(self, log_every=1)
        self.loggers = [self.error_logger, self.config_logger, self.best_model_logger]

    def training_start(self):
        super().training_start()
        for logger in self.loggers:
            logger.training_start()

    def epoch_end(self, **kwargs):
        for logger in self.loggers:
            if self.t.epoch % logger.log_every == 0:
                logger.epoch_end(**kwargs)

    def training_end(self):
        for logger in self.loggers:
            logger.log_summary()

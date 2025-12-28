from .datamodules import *
from .tmr_sku10 import TMRSKU10Dataset

def build_datamodule(args):
    dataset_dict = {
        'SKU10': TMRSKU10DataModule,
    }

    datamodule = dataset_dict[args.dataset](
        args=args,
        datadir=args.datapath,
        batchsize=args.batch_size,
        num_workers=args.num_workers,
        num_exemplars=args.num_exemplars,
    )
    return datamodule

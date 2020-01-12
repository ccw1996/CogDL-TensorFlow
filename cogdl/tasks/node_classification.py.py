import copy
import random

import numpy as np
import tensorflow as tf

from cogdl import options
from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task


@register_task("node_classification")
class NodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        # parser.add_argument("--num-features", type=int)
        # fmt: on

    def __init__(self, args):
        super(NodeClassification, self).__init__(args)

        dataset = build_dataset(args)
        self.data = dataset.data  #假定数据已经从build_dataset中传入
        model = build_model(args) #返回一个keras的model
        self.patience = args.patience
        self.max_epoch = args.max_epoch

    def train(self):
        patience = 0
        best_loss = np.inf
        max_score = 0
        min_loss = np.inf
        for epoch in range (self.max_epoch):
            self.model.fit([self.data.X,self.data.adj],self.data.y_train,batch_size=1,sample_weight=self.data.train_mask,epochs=1,verbose=1,shuffle=False)
            val_loss, val_acc = self.model.evaluate(self.data.X, self.data.adj)
            print("Epoch: {:04d}".format(epoch),"val_loss= {:.4f}".format(val_loss),"val_acc= {:.4f}".formatval_acc))

            val_loss=train_val_loss[1]
            val_acc=train_val_acc[1]
            
            if  val_loss<= min_loss or val_loss>= max_score:
                if val_loss <= best_loss:  # and val_acc >= best_score:
                    best_loss = val_loss
                    best_score = val_acc
                    best_model = copy.deepcopy(self.model)
                min_loss = np.min((min_loss, val_loss))
                max_score = np.max((max_score, val_acc))
                patience = 0
            else:
                if patience == self.patience:
                    print('Epoch {}: early stopping'.format(epoch))
                    break
                patience += 1
        test_loss, test_acc = self.model.evaluate(self.data.y_test, self.data.idx_test)
        return dict(Acc=test_acc)
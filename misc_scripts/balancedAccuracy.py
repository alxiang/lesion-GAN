import tensorflow as tf
import numpy as np


class balancedAccuracy(object):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
    
    def np_balanced_acc(self, y_true, y_pred):
        # Compute the confusion matrix to get the number of true positives,
        # false positives, and false negatives
        # Convert predictions and target from categorical to integer format
        target = np.argmax(y_true, axis=-1).ravel()
        predicted = np.argmax(y_pred, axis=-1).ravel()

        # Trick from torchnet for bincounting 2 arrays together
        # https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        
        # Just in case we get a division by 0, ignore/hide the error and set the value to 0 
        with np.errstate(divide='ignore', invalid='ignore'):
            # normalize to get average recall/specificity
            conf = conf / np.sum(conf, axis=1)[:, np.newaxis]
        conf[np.isnan(conf)] = 0
        
        
        recalls = np.diag(conf)
     
        
        return np.mean(recalls).astype(np.float32)
    
    def sk_balanced_acc(self, y_true, y_pred):
        target = np.argmax(y_true, axis=-1).ravel()
        predicted = np.argmax(y_pred, axis=-1).ravel()
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class = np.diag(conf) / conf.sum(axis=1)
        if np.any(np.isnan(per_class)):
            
            per_class = per_class[~np.isnan(per_class)]
        score = np.mean(per_class)
        
        return score.astype(np.float32)

    def balanced_acc(self, y_true, y_pred):
        # Wraps np_balanced_acc method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self.sk_balanced_acc, [y_true, y_pred], tf.float32)

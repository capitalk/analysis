import numpy as np

class UnivariateThresholder:
  def __init__(self):
    self.threshold = None
    self.left_class = None
    self.right_class = None
    self.classes = None
    
  def fit(self, x, y):
    classes = np.unique(y)
    self.classes = classes
    if len(x.shape == 2):
      x = x[:, 0]
    thresholds = np.unique(x)
    best_threshold = None
    best_left_class = None
    best_right_class = None
    best_acc = 0
    
    for t in thresholds:
      gt = x > t
      lte = ~gt
      left = y[lte]
      right = y[gt]
      left_class = None
      left_count = 0
      for c in classes:
        curr_count = np.sum(left == c)
        if curr_count > left_count:
          left_count = curr_count
          left_class = c
      n_left = float(len(left))
      left_acc = left_count / n_left

      right_count = 0 
      right_class = None
      for c in classes:
        curr_count = np.sum(right == c)
        if curr_count > right_count:
          right_count = curr_count
          right_class = c
      n_right = float(len(right))
      right_acc = right_count / n_right 
      acc = (n_left * left_acc + n_right * right_acc) / (n_left + n_right)
      if acc > best_acc:
        best_acc = acc
        best_threshold = t
        best_left_class = left_class
        best_right_class = right_class
        
    print "thresh = %s, left_class = %s, right_class = %s, with training accuracy = %s" %\
       (best_threshold, best_left_class, best_right_class, best_acc)
    self.threshold = best_threshold  
    self.left_class = best_left_class
    self.right_class = best_right_class
    
  def predict(self, x):
    if len(x.shape == 2):
      x = x[:, 0]
    n = len(x)
    y = np.zeros(n, dtype = self.classes.dtype)
    left_mask = x <= self.threshold
    right_mask = ~left_mask 
    y[left_mask] = self.left_class
    y[right_mask] = self.right_class
    return x
     
          
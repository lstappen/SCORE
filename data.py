class Data:
    def __init__(self, splits, holdout):
        self.splits = splits
        self.split_idx = -1

        # list of numpy arrays, one element = one fold
        self.Xs_train = []
        self.Xs_val = []
        self.ys_train = []
        self.ys_val = []

        if holdout:
            splits -= 1
            self.X_train_holdout = None
            self.X_test_holdout = None
            self.y_train_holdout = None
            self.y_test_holdout = None

    # This could be used as an automatic iter
    def __iter__(self):
        return self

    def __next__(self):
        self.split += 1
        if self.split_idx < self.splits:
            return self.Xs_train[split_idx], self.Xs_val[split_idx], self.ys_train[split_idx], self.ys_val[split_idx]
        raise StopIteration
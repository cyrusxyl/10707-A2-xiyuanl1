# 10707-A2-xiyuanl1
10707 Deep Learning, Homework 2

RBM and AutoEncoder defined as classes in separate files. Each file has its own `__main__` for homework
required functions.

## create learner
RBM `RBM(num_x_units, num_h_units, cd_steps=CD_K)`
AutoEncoder `AutoEncoder(num_x_units, num_h_units)`
Denoising AutoEncoder `AutoEncoder(num_x_units, num_h_units, is_noisy=True)`

## training
```
(train_data, _) = utilities.import_data(filename)
(valid_data, _) = utilities.import_data(filename)
(W, train_loss, valid_loss) = learner.train(train_data_, valid_data_)
```

## plot loss
`utilities.plot_loss(train_loss, valid_loss)`

## plot weights
`utilities.plot_weights(W, num_h_units, num_x_units)`

## pretraining
`utilities.save_weights(filename, W)`
Then load in HW1 Matlab ANN implementation.

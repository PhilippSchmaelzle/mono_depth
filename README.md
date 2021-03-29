

## Evaluation depth values
### Prediction -> Distance
Anelia @google pasted: 
https://github.com/tensorflow/models/issues/6173#issuecomment-471045022

Where she states, that they used the evaluation method from (SfMLearner)[https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py] to do the depth estiation evaluation.

In (this line)[https://github.com/tinghuiz/SfMLearner/blob/2a387b763bc2b6f95b095f929bf751797c9db68a/kitti_eval/eval_depth.py#L66] some scale factor is calculated. The scale factor is applied on the predicted depth image. With this correction there is no direct/constant link between preicted depth and true depth.

For the application of this method, it need to be evaluated if the scale factor varies for each image. If this is **not** the case, it would be possible to calculate an camera and vehicle dependant scale factor and apply it to the predictions.

If it is not constant, the variation of it can be appled to the prediction and encoded within the covariance of the object.
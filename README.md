## CNN architectures:

### Full CNN training
- architecture as used for previous experiments
- replication of basic architecture (before motor inputs) in [1]
- replication of full architecture in [1]

All the above architectures achieve around 0.0002 MSE.

### CNN codes

Achieves 0.002 MSE which is worse than when retraining the whole network. Possibly applying Global Average Pooling could help.


## Conclusions
- drop out reduces performance a lot
- very complex archivectures on CNN codes only improved training MSE

## Comments about data
05_3: no red gear  
10_44: red  gear on a side  

## Future work
- consider lower number of conv layers
- Global Average Pooling for visualising activation maps

## Common issues

Faced this issue: https://github.com/tensorflow/tensorflow/issues/6968
This helped: ```export LD_PRELOAD="/usr/lib/libtcmalloc_minimal.so.4"```

[1] Levine, S., Pastor, P., Krizhevsky, A., Ibarz, J. and Quillen, D., 2016. Learning hand-eye coordination for robotic grasping with deep learning and large-scale data collection. The International Journal of Robotics Research, p.0278364917710318.

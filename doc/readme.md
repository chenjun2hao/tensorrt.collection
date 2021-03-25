##

## 相关问题
### 1. 解决cuda预处理和opencv预处理的结果不一样的问题

- 注意python pytorch 和 tensorrt中都采用opencv进行图像读取和resize操作，然后再转成PIL IMAGE做预处理
```
tensorrt c++ opencv
predict: 309 9.45473
predict: 309 9.56318
predict: 599 9.31222
predict: 304 6.2007
predict: 310 12.2731
predict: 327 8.02724
predict: 310 8.6457

predict: 309 9.45473
predict: 309 9.56318
predict: 599 9.31222
predict: 304 6.2007
predict: 310 12.2731
predict: 327 8.02724
predict: 310 8.6457

tensorrt cuda
predict: 309 9.45901
predict: 309 9.56446
predict: 599 9.29678
predict: 304 6.20385
predict: 310 12.2448
predict: 327 8.03246
predict: 310 8.61893

pytorch
tensor(309) tensor(9.4547, grad_fn=<SelectBackward>) torch.Size([1000]) 0.10304522514343262
tensor(309) tensor(9.5632, grad_fn=<SelectBackward>) torch.Size([1000]) 0.04231905937194824
tensor(599) tensor(9.3122, grad_fn=<SelectBackward>) torch.Size([1000]) 0.07334661483764648
tensor(304) tensor(6.2006, grad_fn=<SelectBackward>) torch.Size([1000]) 0.11889123916625977
tensor(310) tensor(12.2731, grad_fn=<SelectBackward>) torch.Size([1000]) 0.03948354721069336
tensor(327) tensor(8.0272, grad_fn=<SelectBackward>) torch.Size([1000]) 0.08700418472290039
tensor(310) tensor(8.6457, grad_fn=<SelectBackward>) torch.Size([1000]) 0.03231215476989746
```
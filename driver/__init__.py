import torch

OPTIM = {
    'adam': torch.optim.Adam,
    'sgd': torch.optim.SGD,
}
mean = {
    'MosMedData': 0.61346103,
    'COVID-19-CT': 0.62532616,
}
std = {
    'MosMedData': 0.30606743,
    'COVID-19-CT': 0.32330424,
}
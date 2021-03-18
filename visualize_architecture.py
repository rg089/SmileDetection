import imutils
from cvTools.utils.visualize_architecture import visualize_dimensions
from cvTools.ConvNets.LeNet import LeNet

model = LeNet.build(28, 28, 1, 2)
visualize_dimensions(model, "LeNet", path="plots")


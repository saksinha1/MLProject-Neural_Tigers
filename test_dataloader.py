import matplotlib.pyplot as plt
from MNIST_Dataloader import MNIST_Dataloader

def main():
    dataloader = MNIST_Dataloader()
    dataloader.show_images(3, 3)
  
if __name__=="__main__":
    main()
import matplotlib.pyplot as plt
from MNIST_Dataloader import MNIST_Dataloader

def main():
    dataloader = MNIST_Dataloader()
    dataloader.show_images(5, 5)
    # dataloader.simple_show()
  
if __name__=="__main__":
    main()